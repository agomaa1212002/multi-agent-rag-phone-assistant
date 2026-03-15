import ollama

MODEL_NAME = "llama3.1:8b"

def local_generate(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2
) -> str:
    resp = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": temperature}
    )
    return resp["message"]["content"].strip()
