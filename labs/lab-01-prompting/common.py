import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def load_lab_env() -> None:
    lab_dir = Path(__file__).resolve().parent
    env_path = lab_dir / ".env"
    env_example_path = lab_dir / ".env.example"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    elif env_example_path.exists():
        load_dotenv(dotenv_path=env_example_path)


def build_client() -> OpenAI:
    load_lab_env()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY. Add it to .env first.")
    if api_key.startswith("<") or "YOUR_GROQ_API_KEY" in api_key:
        raise ValueError(
            "GROQ_API_KEY is still a placeholder in .env. "
            "Set a real key (gsk_...) in labs/lab-01-prompting/.env."
        )

    base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)

    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        try:
            from langsmith.wrappers import wrap_openai

            client = wrap_openai(client)
            print("[trace] LangSmith tracing enabled")
        except Exception as exc:
            print(f"[warn] LangSmith wrapper unavailable: {exc}")

    return client


def get_model() -> str:
    load_lab_env()
    return os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def run_prompt(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    client = build_client()
    model = get_model()

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""
