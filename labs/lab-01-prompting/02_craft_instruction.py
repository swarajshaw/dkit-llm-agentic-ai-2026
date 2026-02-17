from common import run_prompt


def main() -> None:
    print("\n=== CRAFT prompt ===")
    craft_prompt = (
        "Context: We are preparing lecture notes for postgraduate ML students.\n"
        "Role: You are an experienced ML lecturer who explains concepts clearly with simple analogies.\n"
        "Action: Explain how gradient descent works and why it is used in training neural networks.\n"
        "Format: Provide 6 bullet points followed by a short summary paragraph (80-120 words).\n"
        "Tone: Clear, beginner-friendly, and academically precise."
    )
    print(run_prompt("You follow prompt instructions strictly.", craft_prompt, temperature=0.2))

    print("\n=== Instruction prompting ===")
    instruction_prompt = (
        "Evaluate the feasibility of this project idea by following these steps exactly.\n"
        "Project idea: A mobile app that recommends recipes based on ingredients users already have at home.\n"
        "1. List at least 3 key functional requirements.\n"
        "2. Identify potential technical challenges.\n"
        "3. Suggest suitable technologies/frameworks.\n"
        "4. Provide a 100-150 word feasibility summary.\n"
        "Format with numbered sections matching the steps above."
    )
    print(run_prompt("You are a technical project advisor.", instruction_prompt, temperature=0.2))


if __name__ == "__main__":
    main()
