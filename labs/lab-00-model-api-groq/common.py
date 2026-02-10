import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def load_lab_env():
    # Always load .env from this lab folder, regardless of current working directory.
    lab_dir = Path(__file__).resolve().parent
    env_path = lab_dir / ".env"
    env_example_path = lab_dir / ".env.example"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    elif env_example_path.exists():
        load_dotenv(dotenv_path=env_example_path)


def build_client():
    load_lab_env()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY. Add it to .env first.")

    base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Optional tracing via LangSmith if env vars are configured.
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        try:
            from langsmith.wrappers import wrap_openai

            client = wrap_openai(client)
            print("[trace] LangSmith tracing enabled")
        except Exception as exc:
            print(f"[warn] LangSmith wrapper unavailable: {exc}")

    return client


def get_model():
    load_lab_env()
    return os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
