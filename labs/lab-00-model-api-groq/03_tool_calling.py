import json

from common import build_client, get_model


def get_weather(location: str, unit: str = "celsius"):
    # Mock tool for demo purposes.
    return {
        "location": location,
        "unit": unit,
        "temperature": 6,
        "conditions": "cloudy",
    }


def main():
    client = build_client()
    model = get_model()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather in a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and state, e.g. Chicago, IL",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [
        {"role": "user", "content": "What's the weather like in Chicago today?"}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0,
    )

    msg = response.choices[0].message
    if not msg.tool_calls:
        print("Model returned text without tool call:")
        print(msg.content)
        return

    tool_call = msg.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    tool_result = get_weather(**args)

    messages.append(
        {
            "role": "assistant",
            "tool_calls": [tool_call],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": json.dumps(tool_result),
        }
    )

    final = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )

    print("=== Tool Call Chosen by Model ===")
    print("name:", tool_call.function.name)
    print("args:", args)
    print("\n=== Final Answer ===")
    print(final.choices[0].message.content)


if __name__ == "__main__":
    main()
