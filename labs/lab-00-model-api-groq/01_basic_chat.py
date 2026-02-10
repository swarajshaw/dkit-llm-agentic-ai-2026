from common import build_client, get_model


def main():
    client = build_client()
    model = get_model()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Explain quantum computing in simple terms.",
            }
        ],
        temperature=0.4,
    )

    print("=== Basic Chat Completion ===")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
