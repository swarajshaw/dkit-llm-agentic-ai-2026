import json
from pathlib import Path

from openai import pydantic_function_tool
from pydantic import BaseModel, Field

from common import build_client, get_model


class ReadFileTool(BaseModel):
    """Read a text file from the local lab data directory."""

    name: str = Field(description="File name to read from ./data")

    def run(self) -> str:
        data_dir = Path(__file__).resolve().parent / "data"
        file_path = data_dir / self.name
        try:
            return file_path.read_text(encoding="utf-8")
        except Exception as exc:
            return f"ERROR: {exc}"


def main() -> None:
    client = build_client()
    model = get_model()

    tool_lookup = {"ReadFileTool": ReadFileTool}

    messages = [
        {"role": "system", "content": "You are a useful assistant that reads files."},
        {
            "role": "user",
            "content": "Please open the file build_agents_extract.txt and read its content.",
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        tools=[pydantic_function_tool(ReadFileTool)],
    )

    tool_call = response.choices[0].message.tool_calls[0]
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)

    file_content = tool_lookup[tool_name](**tool_args).run()

    summary_messages = [
        {"role": "system", "content": "You are a useful assistant that summarizes text."},
        {
            "role": "user",
            "content": f"Please summarize the following text:\n\n{file_content}",
        },
    ]

    summary_response = client.chat.completions.create(
        model=model,
        messages=summary_messages,
        temperature=0.2,
    )

    print("\n=== File Content (truncated) ===\n")
    print(file_content[:700])
    print("\n=== Summary ===\n")
    print(summary_response.choices[0].message.content)


if __name__ == "__main__":
    main()
