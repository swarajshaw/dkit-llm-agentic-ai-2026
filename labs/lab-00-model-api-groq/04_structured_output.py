import json

from pydantic import BaseModel, Field, ValidationError

from common import build_client, get_model


class City(BaseModel):
    name: str = Field(description="The name of the city")
    country: str = Field(description="The country where the city is located")
    population: int = Field(description="Approximate population of the city")


def main():
    client = build_client()
    model = get_model()

    system_prompt = (
        "Return only valid JSON with keys: name, country, population. "
        "population must be an integer."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Provide details about Paris."},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
    except Exception:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )

    content = response.choices[0].message.content
    try:
        data = json.loads(content)
        city = City.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        print("Structured parse failed.")
        print("Raw model output:")
        print(content)
        print("\nError:")
        print(exc)
        return

    print("=== Structured Output (Validated) ===")
    print(city.model_dump())


if __name__ == "__main__":
    main()
