from common import build_client, get_model


def chat(client, model: str, system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""


def main() -> None:
    client = build_client()
    model = get_model()

    question = "How could AI help reduce traffic congestion in large cities?"

    step_1 = chat(
        client,
        model,
        "You identify root causes clearly.",
        f"Break down this problem into 5 key causes only. Question: {question}",
    )
    print("\n=== Step 1: Causes ===\n")
    print(step_1)

    step_2 = chat(
        client,
        model,
        "You design practical AI interventions.",
        "For each cause below, suggest one AI-based solution in bullet points.\n\n"
        f"Causes:\n{step_1}",
    )
    print("\n=== Step 2: AI Solutions ===\n")
    print(step_2)

    step_3 = chat(
        client,
        model,
        "You synthesize outputs into decision-ready summaries.",
        "Write one concise paragraph recommending the most promising approach and why, "
        "using the outputs below.\n\n"
        f"Causes:\n{step_1}\n\nSolutions:\n{step_2}",
    )
    print("\n=== Step 3: Final Synthesis ===\n")
    print(step_3)


if __name__ == "__main__":
    main()
