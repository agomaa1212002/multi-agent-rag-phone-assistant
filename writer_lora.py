# writer_lora.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

SYSTEM_PROMPT = (
    "You are a phone product assistant. "
    "Answer ONLY using the provided phone data. "
    "If information is missing, say \"I don't have that information.\" "
    "Be concise, factual, and user-friendly."
)

class LoraWriter:
    """
    LoRA-tuned HuggingFace writer.
    Loads base model once, applies LoRA adapter, then generates answers.
    """

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-3B-Instruct",
        adapter_dir: str = r"C:\LLM_PROJECTR\lora_out",
        max_new_tokens: int = 220,
        temperature: float = 0.6,   # 🔥 ADD THIS
        top_p: float = 0.9          # 🔥 ADD THIS
    ):
        self.base_model_name = base_model
        self.adapter_dir = adapter_dir
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.tok = AutoTokenizer.from_pretrained(
            self.base_model_name,
            use_fast=True
        )

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map="auto",
            torch_dtype="auto",
        )

        self.model = PeftModel.from_pretrained(base, self.adapter_dir)
        self.model.eval()

    def _build_prompt(self, user_text: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        # Qwen chat template
        return self.tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    @torch.no_grad()
    def write(self, user_text: str) -> str:
        prompt = self._build_prompt(user_text)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)

        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,                # ✅ REQUIRED
            temperature=self.temperature,  # ✅ ENABLED
            top_p=self.top_p,              # ✅ ENABLED
            top_k=None
        )

        gen_tokens = out[0][inputs["input_ids"].shape[-1]:]
        return self.tok.decode(
            gen_tokens,
            skip_special_tokens=True
        ).strip()
