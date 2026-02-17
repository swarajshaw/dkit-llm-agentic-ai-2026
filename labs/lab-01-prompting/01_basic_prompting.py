from common import run_prompt


def main() -> None:
    print("\n=== Zero-shot ===")
    zero_shot = (
        "Classify the sentiment of this review as positive, negative, or neutral. "
        "Return only one word.\n"
        "Review: The movie was okay, nothing special but not terrible either."
    )
    print(run_prompt("You are a concise classifier.", zero_shot, temperature=0.0))

    print("\n=== Single-shot ===")
    single_shot = (
        "Convert sentence to passive voice.\n"
        "Example:\n"
        "Active: The chef prepared the meal.\n"
        "Passive: The meal was prepared by the chef.\n\n"
        "Now convert:\n"
        "Active: The students completed the assignment."
    )
    print(run_prompt("You are a grammar assistant.", single_shot, temperature=0.0))

    print("\n=== Few-shot ===")
    few_shot = (
        "Extract key information from each review and return JSON with keys: "
        "sentiment, key_features, concerns, rating_implied.\n\n"
        "Example 1\n"
        "Review: Great phone! Battery lasts all day and camera quality is amazing. A bit pricey though.\n"
        "Output: {\"sentiment\": \"positive\", \"key_features\": [\"battery life\", \"camera quality\"], \"concerns\": [\"price\"], \"rating_implied\": \"high\"}\n\n"
        "Example 2\n"
        "Review: Terrible service. Food was cold and waiter was rude. Never coming back.\n"
        "Output: {\"sentiment\": \"negative\", \"key_features\": [], \"concerns\": [\"service quality\", \"food temperature\", \"staff behavior\"], \"rating_implied\": \"low\"}\n\n"
        "Now process:\n"
        "Review: Love this restaurant! The pasta was perfectly cooked and the ambiance was romantic. Only downside was long wait time."
    )
    print(run_prompt("You are an information extraction assistant.", few_shot, temperature=0.1))


if __name__ == "__main__":
    main()
