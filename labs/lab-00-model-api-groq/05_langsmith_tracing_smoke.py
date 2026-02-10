import os

from common import build_client, get_model, load_lab_env


def main():
    load_lab_env()
    if os.getenv("LANGSMITH_TRACING", "").lower() != "true":
        print(
            "Set LANGSMITH_TRACING=true in .env (or .env.example) "
            "to run this tracing smoke test."
        )
        return

    client = build_client()
    model = get_model()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are concise.",
            },
            {
                "role": "user",
                "content": "Give me one sentence about why tracing helps debugging.",
            },
        ],
        temperature=0.1,
    )

    print("=== LangSmith Tracing Smoke Test ===")
    print(response.choices[0].message.content)
    print("If tracing env vars are correct, this call should appear in LangSmith.")


if __name__ == "__main__":
    main()
