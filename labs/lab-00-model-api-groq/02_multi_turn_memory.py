from common import build_client, get_model


def ask(client, model, history):
    response = client.chat.completions.create(
        model=model,
        messages=history,
        temperature=0.2,
    )
    text = response.choices[0].message.content
    history.append({"role": "assistant", "content": text})
    return text


def main():
    client = build_client()
    model = get_model()

    history = [
        {
            "role": "system",
            "content": "You are a clear Python tutor. Keep answers concise.",
        }
    ]

    history.append({"role": "user", "content": "What is a Python list?"})
    print("Q1:", history[-1]["content"])
    print("A1:", ask(client, model, history))
    print()

    history.append({"role": "user", "content": "How do I add items to it?"})
    print("Q2:", history[-1]["content"])
    print("A2:", ask(client, model, history))
    print()

    history.append(
        {
            "role": "user",
            "content": "What is the difference between append() and extend()?",
        }
    )
    print("Q3:", history[-1]["content"])
    print("A3:", ask(client, model, history))


if __name__ == "__main__":
    main()
