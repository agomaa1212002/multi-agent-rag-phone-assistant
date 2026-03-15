# multi_agent_llm_planner.py
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from retrieve import retrieve
from tools import load_phones, filter_phones, top_battery, compare_phones

# Planner + Verifier use local LLM (Ollama) via your existing wrapper
from local_llm import local_generate

# Writer uses LoRA-tuned HuggingFace model
from writer_lora import LoraWriter


phones = load_phones()

SYSTEM_PROMPT = (
    "You are a phone product assistant. "
    "Answer ONLY using the provided phone data. "
    'If information is missing, say "I don\'t have that information." '
    "Be concise, factual, and user-friendly."
)

PLANNER_PROMPT = """
You are a planner agent for a phone assistant.
Your job: output ONLY valid JSON (no extra text) with this schema:

{
  "mode": "RAG" | "TOOL" | "BOTH" | "REFUSE",
  "tool_name": "filter_phones" | "top_battery" | "compare_phones" | null,
  "tool_args": object | null,
  "rag_query": string | null,
  "top_k": integer,
  "reason": string
}

Rules:
- If tool_name is "filter_phones", tool_args must use numeric fields:
  min_ram_gb (number), min_storage_gb (number), min_battery_mah (integer),
  os_contains (string or null), brand (string or null), limit (integer or null)
- Do NOT use keys like "ram", "storage", "battery". Use min_* fields instead.
- If the query includes price/budget constraints (under $..., price, budget) choose "REFUSE".
- Use "compare_phones" for comparisons (compare / vs / versus) and tool_args MUST include:
  {"name_a": "<phone A>", "name_b": "<phone B>"}.
  If you cannot extract both, return {"mode":"REFUSE"}.
- Use "top_battery" for best/highest/top battery requests.
- Use "filter_phones" when user gives constraints about RAM, storage, battery, OS.
- Use "RAG" when user asks for information about a specific phone.
- Use "BOTH" when filtering is needed plus RAG details help.
- If user mentions Android, set os_contains to "android".
Return JSON only.
""".strip()

VERIFIER_PROMPT = """
You are a strict verifier/critic for a phone assistant.

You will be given:
1) User question
2) Context (retrieved/tool text)
3) Draft answer

Rules:
- You MUST enforce: answer ONLY using facts that appear in Context.
- If Context contains relevant facts, the answer MUST NOT say "I don't have that information."
- If Context does NOT contain enough facts to answer, final answer must be exactly
- If user asks about price/budget/cost (price, $, under, cheap, affordable, value for money),
  final answer must be exactly:
  I don't have that information.
- Remove hallucinations: if draft includes facts not in context, rewrite without them.
- If draft refuses but context contains facts, request rewrite.

Output ONLY valid JSON in one of these forms (no extra text):
{"decision":"accept","final_answer":"..."}
{"decision":"rewrite","rewrite_instructions":"..."}
{"decision":"refuse","final_answer":"I don't have that information."}
""".strip()


@dataclass
class Plan:
    mode: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    rag_query: Optional[str] = None
    top_k: int = 3
    reason: str = ""


# ----------------------------
# Helpers
# ----------------------------
def _is_price_query(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in [
        "price", "budget", "$", "under", "cheap", "cheapest",
        "affordable", "cost", "value for money", "expensive"
    ])

def _safe_int(x, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default

def _extract_compare_names(query: str) -> Optional[List[str]]:
    q = query.strip()
    if re.search(r"\bvs\b", q, flags=re.IGNORECASE):
        parts = re.split(r"\bvs\b", q, flags=re.IGNORECASE)
    elif re.search(r"\bversus\b", q, flags=re.IGNORECASE):
        parts = re.split(r"\bversus\b", q, flags=re.IGNORECASE)
    else:
        parts = re.split(r"\bcompare\b|\band\b", q, flags=re.IGNORECASE)

    parts = [p.strip(" :-") for p in parts if p.strip(" :-")]
    if len(parts) >= 2:
        return [parts[0], parts[1]]
    return None

def format_compare(a: dict, b: dict) -> str:
    fields = [
        "operating_system",
        "chipset",
        "display_type",
        "ram",
        "internal_storage",
        "battery_capacity",
        "quick_charging",
        "primary_camera_resolution",
        "refresh_rate",
    ]
    lines = []
    for f in fields:
        av = a.get(f)
        bv = b.get(f)
        if av == bv:
            lines.append(f"- {f}: same ({av})")
        else:
            lines.append(f"- {f}: A={av} | B={bv}")
    return "\n".join(lines)


# ----------------------------
# LLM Planner agent
# ----------------------------
class LLMPlannerAgent:
    def plan(self, user_query: str) -> Plan:
        # Hard safety: price/budget → refuse (even if planner fails)
        if _is_price_query(user_query):
            return Plan(mode="REFUSE", reason="Query includes price/budget constraints")

        raw = local_generate(
            system_prompt=PLANNER_PROMPT,
            user_prompt=f"User query: {user_query}",
            temperature=0.0
        )

        try:
            plan_dict = json.loads(raw)
        except json.JSONDecodeError:
            return Plan(mode="RAG", rag_query=user_query, top_k=3, reason="Planner JSON parse failed → fallback RAG")

        return Plan(
            mode=plan_dict.get("mode", "RAG"),
            tool_name=plan_dict.get("tool_name"),
            tool_args=plan_dict.get("tool_args"),
            rag_query=plan_dict.get("rag_query"),
            top_k=_safe_int(plan_dict.get("top_k") or 3, 3),
            reason=plan_dict.get("reason", "")
        )


# ----------------------------
# Writer agent (LoRA)
# ----------------------------
class WriterAgent:
    def __init__(self):
        self.lora = LoraWriter()

    def write(self, user_query: str, context: str, extra_instructions: str = "") -> str:
        user_text = f"""
Phone Data:
{context}

User Question:
{user_query}

Instructions:
- Answer in natural, fluent English.
- Use ONLY the Phone Data above.
- Do NOT guess or add missing specs.
- If required information is missing, reply exactly:
  I don't have that information.
{extra_instructions}

Answer:
""".strip()

        return self.lora.write(user_text)


# ----------------------------
# Verifier agent (Ollama/local LLM)
# ----------------------------
class VerifierAgent:
    """
    Uses local_generate (Ollama) to enforce correctness + refusal policy.
    Deterministic JSON output.
    """
    def verify(self, user_query: str, context: str, draft_answer: str) -> Dict[str, str]:
        # Hard safety: refuse price always
        if _is_price_query(user_query):
            return {"decision": "refuse", "final_answer": "I don't have that information."}

        payload = f"""
User Question:
{user_query}

Context:
{context}

Draft Answer:
{draft_answer}
""".strip()

        raw = local_generate(
            system_prompt=VERIFIER_PROMPT,
            user_prompt=payload,
            temperature=0.0
        )

        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            # Safe fallback: accept draft unless it contains refusal while context has data
            return {"decision": "accept", "final_answer": draft_answer.strip()}

        decision = (obj.get("decision") or "").lower().strip()
        if decision not in ("accept", "rewrite", "refuse"):
            return {"decision": "accept", "final_answer": draft_answer.strip()}

        if decision == "accept":
            return {"decision": "accept", "final_answer": str(obj.get("final_answer") or draft_answer).strip()}

        if decision == "refuse":
            return {"decision": "refuse", "final_answer": "I don't have that information."}

        # rewrite
        return {
            "decision": "rewrite",
            "rewrite_instructions": str(obj.get("rewrite_instructions") or "").strip()
        }


# ----------------------------
# Executor agent
# ----------------------------
class ExecutorAgent:
    def __init__(self):
        self.writer = WriterAgent()
        self.verifier = VerifierAgent()

    def run(self, user_query: str, plan: Plan) -> Tuple[str, Dict[str, Any]]:
        """
        Returns: (final_answer, debug_info)
        """
        debug_info: Dict[str, Any] = {
            "context_type": None,
            "context_preview": None,
            "verifier": None,
        }

        # Refusal
        if plan.mode == "REFUSE":
            debug_info["context_type"] = "none"
            debug_info["verifier"] = {"decision": "refuse", "final_answer": "I don't have that information."}
            return "I don't have that information.", debug_info

        # Build context depending on plan
        context = ""

        if plan.mode == "RAG":
            docs = retrieve(plan.rag_query or user_query, top_k=plan.top_k)
            debug_info["context_type"] = "rag"
            context = "\n".join(docs).strip()
            if not context:
                debug_info["verifier"] = {"decision": "refuse", "final_answer": "I don't have that information."}
                return "I don't have that information.", debug_info

        elif plan.mode == "TOOL":
            tool_out = self._run_tool(plan.tool_name, plan.tool_args, user_query)
            debug_info["context_type"] = f"tool:{plan.tool_name}"
            if tool_out is None:
                debug_info["verifier"] = {"decision": "refuse", "final_answer": "I don't have that information."}
                return "I don't have that information.", debug_info
            context = self._tool_to_text(plan.tool_name, tool_out).strip()
            if not context:
                debug_info["verifier"] = {"decision": "refuse", "final_answer": "I don't have that information."}
                return "I don't have that information.", debug_info

        elif plan.mode == "BOTH":
            tool_out = self._run_tool(plan.tool_name, plan.tool_args, user_query)
            debug_info["context_type"] = f"both:{plan.tool_name}"
            if tool_out is None:
                debug_info["verifier"] = {"decision": "refuse", "final_answer": "I don't have that information."}
                return "I don't have that information.", debug_info

            tool_context = self._tool_to_text(plan.tool_name, tool_out).strip()

            rag_context = ""
            if isinstance(tool_out, list) and tool_out:
                names = [
                    f"{p.get('brand','')} {p.get('model','')}".strip()
                    for p in tool_out[:3]
                ]
                docs = []
                for n in names:
                    docs.extend(retrieve(f"info about {n}", top_k=1))
                rag_context = "\n".join(docs).strip()

            context = tool_context + ("\n\nRAG Details:\n" + rag_context if rag_context else "")
            context = context.strip()

            if not context:
                debug_info["verifier"] = {"decision": "refuse", "final_answer": "I don't have that information."}
                return "I don't have that information.", debug_info

        else:
            debug_info["context_type"] = "unknown"
            debug_info["verifier"] = {"decision": "refuse", "final_answer": "I don't have that information."}
            return "I don't have that information.", debug_info

        debug_info["context_preview"] = (context[:900] + "…") if len(context) > 900 else context

        # ----------------------------
        # Writer -> Verifier -> (optional one rewrite)
        # ----------------------------
        draft = self.writer.write(user_query, context)

        v1 = self.verifier.verify(user_query, context, draft)
        debug_info["verifier"] = {"first_pass": v1}

        if v1["decision"] == "accept":
            return v1["final_answer"], debug_info

        if v1["decision"] == "refuse":
            return "I don't have that information.", debug_info

        # rewrite once
        rewrite_instructions = v1.get("rewrite_instructions", "").strip()
        extra = f"""
Verifier rewrite requirements:
- Do NOT refuse if the context contains relevant specs.
- Remove any facts not present in Phone Data.
- Keep it concise but natural.
{rewrite_instructions}
""".strip()

        draft2 = self.writer.write(user_query, context, extra_instructions=extra)
        v2 = self.verifier.verify(user_query, context, draft2)
        debug_info["verifier"]["second_pass"] = v2

        if v2["decision"] == "accept":
            return v2["final_answer"], debug_info

        if v2["decision"] == "refuse":
            return "I don't have that information.", debug_info

        # If still wants rewrite, accept draft2 as best-effort (avoid infinite loop)
        return draft2.strip(), debug_info

    def _run_tool(self, tool_name: Optional[str], tool_args: Optional[Dict[str, Any]], user_query: str):
        if tool_name is None:
            return None

        args = tool_args or {}

        # Normalize: if planner produced "os" use "os_contains"
        if "os" in args and "os_contains" not in args:
            args["os_contains"] = args.pop("os")

        # Remove None values
        args = {k: v for k, v in args.items() if v is not None}

        # If planner forgot os_contains but user clearly said android/windows
        ql = user_query.lower()
        if "android" in ql and "os_contains" not in args:
            args["os_contains"] = "android"
        if "windows" in ql and "os_contains" not in args:
            args["os_contains"] = "windows"

        # Convert string constraints if planner makes mistakes
        if "ram" in args and "min_ram_gb" not in args:
            m = re.search(r"(\d+)", str(args["ram"]))
            if m:
                args["min_ram_gb"] = float(m.group(1))
            args.pop("ram", None)

        if "storage" in args and "min_storage_gb" not in args:
            m = re.search(r"(\d+)", str(args["storage"]))
            if m:
                args["min_storage_gb"] = float(m.group(1))
            args.pop("storage", None)

        if "battery" in args and "min_battery_mah" not in args:
            m = re.search(r"(\d+)", str(args["battery"]))
            if m:
                args["min_battery_mah"] = int(m.group(1))
            args.pop("battery", None)

        # compare tool
        if tool_name == "compare_phones":
            if "name_a" not in args or "name_b" not in args:
                names = _extract_compare_names(user_query)
                if not names:
                    return None
                args["name_a"], args["name_b"] = names[0], names[1]

            out = compare_phones(phones, args["name_a"], args["name_b"])
            return None if isinstance(out, dict) and "error" in out else out

        # filter tool
        if tool_name == "filter_phones":
            return filter_phones(phones, **args)

        # top battery tool
        if tool_name == "top_battery":
            allowed = {"brand", "os_contains", "top_k"}
            safe_args = {k: v for k, v in args.items() if k in allowed}
            return top_battery(phones, **safe_args)

        return None

    def _tool_to_text(self, tool_name: str, tool_out: Any) -> str:
        if tool_name == "top_battery":
            return "\n".join([f"{name}: {mah} mAh" for name, mah in tool_out])

        if tool_name == "compare_phones":
            a = tool_out["A"]
            b = tool_out["B"]
            header = f"Comparison: {a.get('brand','')} {a.get('model','')} vs {b.get('brand','')} {b.get('model','')}"
            return header + "\n" + format_compare(a, b)

        if tool_name == "filter_phones":
            if not tool_out:
                return ""
            lines = []
            for p in tool_out:
                lines.append(
                    f"{p.get('brand','')} {p.get('model','')}: "
                    f"OS={p.get('operating_system')}, RAM={p.get('ram')}, "
                    f"Storage={p.get('internal_storage')}, Battery={p.get('battery_capacity')}, "
                    f"Camera={p.get('primary_camera_resolution')}, Refresh={p.get('refresh_rate')}"
                )
            return "\n".join(lines)

        return str(tool_out)


# ----------------------------
# Orchestrator
# ----------------------------
class MultiAgentAssistant:
    def __init__(self):
        self.planner = LLMPlannerAgent()
        self.executor = ExecutorAgent()

    def chat(self, user_query: str, debug: bool = False) -> str:
        plan = self.planner.plan(user_query)
        ans, dbg = self.executor.run(user_query, plan)

        if debug:
            verifier_block = dbg.get("verifier", {})
            return (
                f"[Planner decision]\n"
                f"- mode: {plan.mode}\n"
                f"- tool: {plan.tool_name}\n"
                f"- tool_args: {plan.tool_args}\n"
                f"- rag_query: {plan.rag_query}\n"
                f"- top_k: {plan.top_k}\n"
                f"- reason: {plan.reason}\n\n"
                f"[Context]\n"
                f"- type: {dbg.get('context_type')}\n"
                f"- preview:\n{dbg.get('context_preview')}\n\n"
                f"[Verifier]\n{json.dumps(verifier_block, ensure_ascii=False, indent=2)}\n\n"
                f"[Assistant]\n{ans}"
            )

        return ans
