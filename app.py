import streamlit as st
import pandas as pd

from retrieve import retrieve
from multi_agent_llm_planner import MultiAgentAssistant


# ----------------------------
# Page config (polished look)
# ----------------------------
st.set_page_config(
    page_title="LLM_PROJECTR Phone Assistant",
    page_icon="",
    layout="wide",
)

st.title("  Phone Product Assistant")
st.caption("Multi-agent (Planner + Tools + RAG) with LoRA-tuned local writer")


# ----------------------------
# Load assistant once
# ----------------------------
@st.cache_resource
def load_bot():
    return MultiAgentAssistant()

bot = load_bot()


# ----------------------------
# Session state: chat history
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_debug" not in st.session_state:
    st.session_state.last_debug = None


# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider("RAG top_k", min_value=1, max_value=8, value=3, step=1)
    show_debug = st.toggle("Show planner + context", value=True)

    st.divider()
    st.subheader("🧠 Agent Panel")

    if st.session_state.last_debug:
        dbg = st.session_state.last_debug

        st.markdown("**Planner Decision**")
        st.code(dbg["plan_pretty"], language="json")

        st.markdown("**Mode / Tool**")
        st.write(f"Mode: **{dbg['mode']}**")
        st.write(f"Tool: **{dbg.get('tool_name')}**")

        if dbg.get("tool_args") is not None:
            st.markdown("**Tool Args**")
            st.json(dbg["tool_args"])

        if dbg.get("rag_docs"):
            st.markdown("**Retrieved RAG Docs**")
            for i, d in enumerate(dbg["rag_docs"], 1):
                with st.expander(f"Doc {i}", expanded=False):
                    st.write(d)

        if dbg.get("tool_view") is not None:
            st.markdown("**Tool Output**")
            tool_view = dbg["tool_view"]

            # If it's a table
            if isinstance(tool_view, pd.DataFrame):
                st.dataframe(tool_view, use_container_width=True)
            # Otherwise text
            else:
                st.write(tool_view)

    else:
        st.info("Ask a question to see planner decision, RAG docs, and tool output.")

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_debug = None
        st.rerun()


# ----------------------------
# Display chat history
# ----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ----------------------------
# Helper: pretty plan
# ----------------------------
def plan_to_pretty_json(plan) -> str:
    obj = {
        "mode": plan.mode,
        "tool_name": plan.tool_name,
        "tool_args": plan.tool_args,
        "rag_query": plan.rag_query,
        "top_k": plan.top_k,
        "reason": plan.reason,
    }
    import json
    return json.dumps(obj, indent=2, ensure_ascii=False)


# ----------------------------
# Helper: render tool outputs
# ----------------------------
def tool_output_to_view(plan, tool_out):
    """
    Return either a DataFrame (nice table) or a short text for sidebar display.
    """
    if tool_out is None:
        return None

    if plan.tool_name == "top_battery":
        # tool_out is list of (name, mah)
        df = pd.DataFrame(tool_out, columns=["Phone", "Battery (mAh)"])
        return df

    if plan.tool_name == "filter_phones":
        # tool_out is list[dict]
        if isinstance(tool_out, list) and tool_out:
            rows = []
            for p in tool_out[:20]:
                rows.append({
                    "Phone": f"{p.get('brand','')} {p.get('model','')}".strip(),
                    "OS": p.get("operating_system"),
                    "RAM": p.get("ram"),
                    "Storage": p.get("internal_storage"),
                    "Battery": p.get("battery_capacity"),
                    "Camera": p.get("primary_camera_resolution"),
                    "Refresh": p.get("refresh_rate"),
                })
            return pd.DataFrame(rows)
        return "No matches."

    if plan.tool_name == "compare_phones":
        # tool_out is {"A":..., "B":...}
        if isinstance(tool_out, dict) and "A" in tool_out and "B" in tool_out:
            a = tool_out["A"]; b = tool_out["B"]
            fields = [
                "operating_system","chipset","ram","internal_storage",
                "battery_capacity","primary_camera_resolution","refresh_rate"
            ]
            rows = []
            for f in fields:
                rows.append({
                    "Field": f,
                    "A": a.get(f),
                    "B": b.get(f),
                    "Same?": a.get(f) == b.get(f),
                })
            return pd.DataFrame(rows)
        return "Comparison failed."

    # fallback
    return str(tool_out)


# ----------------------------
# Main chat input
# ----------------------------
user_query = st.chat_input("Ask about phones (info, compare, best battery, filter specs)...")

if user_query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Planner decision (LLM planner)
    plan = bot.planner.plan(user_query)

    # Override top_k from UI slider
    plan.top_k = top_k

    rag_docs = []
    tool_out = None
    tool_context = ""
    combined_context = ""

    # Execute according to plan, while capturing context for UI
    if plan.mode == "REFUSE":
        answer = "I don't have that information."
    else:
        # RAG
        if plan.mode in ("RAG", "BOTH"):
            rag_docs = retrieve(plan.rag_query or user_query, top_k=plan.top_k) or []

        # TOOL
        if plan.mode in ("TOOL", "BOTH"):
            tool_out = bot.executor._run_tool(plan.tool_name, plan.tool_args, user_query)
            if tool_out is not None:
                tool_context = bot.executor._tool_to_text(plan.tool_name, tool_out)

        # Build final context passed into LoRA writer
        if plan.mode == "RAG":
            combined_context = "\n".join(rag_docs).strip()
        elif plan.mode == "TOOL":
            combined_context = tool_context.strip()
        elif plan.mode == "BOTH":
            combined_context = (tool_context + "\n\nRAG Details:\n" + "\n".join(rag_docs)).strip()
        else:
            combined_context = ""

        if not combined_context:
            answer = "I don't have that information."
        else:
            # Use the SAME writer your pipeline uses (LoRA Qwen)
            answer = bot.executor.writer.write(user_query, combined_context)

    # Save debug info for sidebar
    if show_debug:
        st.session_state.last_debug = {
            "mode": plan.mode,
            "tool_name": plan.tool_name,
            "tool_args": plan.tool_args,
            "rag_docs": rag_docs,
            "tool_out": tool_out,
            "tool_view": tool_output_to_view(plan, tool_out),
            "plan_pretty": plan_to_pretty_json(plan),
        }
    else:
        st.session_state.last_debug = None

    # Show assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.rerun()
