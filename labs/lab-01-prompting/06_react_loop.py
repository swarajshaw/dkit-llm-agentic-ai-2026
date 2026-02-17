import json
from pathlib import Path
from typing import Sequence

from openai import pydantic_function_tool
from pydantic import BaseModel, Field

from common import build_client, get_model


class ListFilesTool(BaseModel):
    """List text files in the local lab data directory."""

    folder: str = Field(default="data", description="Folder name relative to this script")

    def run(self) -> str:
        base = Path(__file__).resolve().parent
        target = base / folder_safe(self.folder)
        try:
            files = sorted([p.name for p in target.glob("*.txt")])
            return json.dumps(files)
        except Exception as exc:
            return f"ERROR: {exc}"


class ReadFileTool(BaseModel):
    """Read one text file from the local lab data directory."""

    name: str = Field(description="File name inside ./data")

    def run(self) -> str:
        base = Path(__file__).resolve().parent
        file_path = base / "data" / self.name
        try:
            return file_path.read_text(encoding="utf-8")
        except Exception as exc:
            return f"ERROR: {exc}"


def folder_safe(folder: str) -> str:
    return folder.replace("..", "").strip("/") or "data"


def execute_function(tool_call, tool_lookup):
    function_name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    tool = tool_lookup[function_name](**args)
    return function_name, tool.run()


def react_loop(messages, client, model: str, tools: Sequence[type[BaseModel]]):
    tool_lookup = {tool.__name__: tool for tool in tools}
    tool_schemas = [pydantic_function_tool(tool) for tool in tools]

    while True:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=700,
            tools=tool_schemas,
        )

        assistant_msg = response.choices[0].message
        tools_to_run = assistant_msg.tool_calls

        if not tools_to_run:
            messages.append({"role": "assistant", "content": assistant_msg.content})
            return assistant_msg.content

        for tool_call in tools_to_run:
            function_name, result = execute_function(tool_call, tool_lookup)
            messages.append(
                {
                    "role": "tool",
                    "name": function_name,
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )


def main() -> None:
    client = build_client()
    model = get_model()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a useful assistant that reads files and creates combined summaries. "
                "First list files in the folder, then read each file, then produce a single concise summary."
            ),
        },
        {
            "role": "user",
            "content": "Read all .txt files in the data folder and produce one combined summary.",
        },
    ]

    final = react_loop(messages, client, model, [ListFilesTool, ReadFileTool])
    print("\n=== ReAct Final Summary ===\n")
    print(final)


if __name__ == "__main__":
    main()
