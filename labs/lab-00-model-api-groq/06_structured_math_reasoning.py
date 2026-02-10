from pydantic import BaseModel

from common import build_client, get_model


class Step(BaseModel):
    explanation: str
    output: str


class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str


def main():
    client = build_client()
    model = get_model()

    completion = client.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful math tutor. "
                    "Guide the user through the solution step by step."
                ),
            },
            {"role": "user", "content": "How can I solve 8x + 7 = -23?"},
        ],
        response_format=MathReasoning,
    )

    solution = completion.choices[0].message.parsed

    print("=== Structured Math Reasoning ===")
    for step_number, step in enumerate(solution.steps, start=1):
        print(f"Step {step_number}: {step.explanation}")
        print(f"Output: {step.output}\n")
    print("Final Answer:", solution.final_answer)


if __name__ == "__main__":
    main()
