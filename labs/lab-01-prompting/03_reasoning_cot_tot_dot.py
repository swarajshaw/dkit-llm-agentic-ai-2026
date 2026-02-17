from common import run_prompt


def main() -> None:
    print("\n=== Chain of Thought (CoT) ===")
    cot_prompt = (
        "Solve this step by step and show intermediate calculations.\n"
        "A company's revenue increased by 15% in Q1, decreased by 8% in Q2, "
        "and increased by 12% in Q3. Initial revenue was $100,000. "
        "What is the revenue at the end of Q3?"
    )
    print(run_prompt("You are a careful quantitative reasoning assistant.", cot_prompt, temperature=0.0))

    print("\n=== Tree of Thought (ToT) ===")
    tot_prompt = (
        "Plan a 6-month research project timeline with literature review, data collection, analysis, and writing. "
        "I have 2 research assistants for 3 months.\n"
        "Provide 3 approaches: sequential, parallel, hybrid.\n"
        "For each: pros, cons, timeline estimate.\n"
        "Then recommend the best approach with justification."
    )
    print(run_prompt("You are a research planning advisor.", tot_prompt, temperature=0.2))

    print("\n=== Draft of Thought (DoT) ===")
    dot_prompt = (
        "Write a professional email to a potential research collaborator.\n"
        "Context: PhD student contacting a professor in sustainable urban planning.\n"
        "Step 1: Write an initial draft.\n"
        "Step 2: Critique draft (tone, clarity, persuasiveness, professionalism).\n"
        "Step 3: Write improved final version."
    )
    print(run_prompt("You are a scientific communication assistant.", dot_prompt, temperature=0.3))


if __name__ == "__main__":
    main()
