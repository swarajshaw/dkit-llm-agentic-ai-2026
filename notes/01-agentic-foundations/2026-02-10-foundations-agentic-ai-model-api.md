# 2026-02-10 Foundations in Agentic AI - Interacting with Model API

## Lecture objective

-   Learn how to communicate with a language model through API calls.
-   Understand chat interaction patterns and message roles.
-   Make basic model calls and extend them with chat history.
-   Introduce tool calling and structured outputs.
-   Get a first observability workflow with LangSmith tracing.

## What was taught

-   API communication model:
    -   HTTP-based REST endpoints
    -   JSON request/response payloads
    -   stateless protocol (every call is independent)
    -   token usage as a core cost/latency unit
-   Model landscape overview:
    -   cloud LLMs (for example ChatGPT, Claude, Gemini, DeepSeek, Groq)
    -   local SLMs (for example Llama, Qwen, Mistral) via tools such as Ollama
-   OpenAI SDK as a low-level interface:
    -   create a client with `api_key` and `base_url`
    -   call `client.chat.completions.create(...)`
    -   read output from `response.choices[0].message.content`
-   API libraries vs frameworks:
    -   API library = thin wrapper over HTTP model calls
    -   framework = adds orchestration/state/agent abstractions
    -   for simple intelligence needs, API libraries reduce complexity

### Quick example: basic API call

``` python
from openai import OpenAI

client = OpenAI(api_key="...", base_url="https://api.groq.com/openai/v1")
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Explain quantum computing simply."}],
    temperature=0.4,
)
print(response.choices[0].message.content)
```

## Model call essentials

-   Required parameters:
    -   `api_key`: Secret token used to authenticate your application with the model provider.
    -   `model`: The exact model ID/version the API should execute.
    -   `messages`: Ordered conversation context (system/user/assistant) passed as input.
-   Common controls:
    -   `temperature`: Controls randomness; lower is more deterministic, higher is more creative.
    -   `top_p`: Uses nucleus sampling to limit token choices to a probability mass.
    -   `max_tokens`: Sets the maximum number of tokens in the model output.
    -   `stream`: Returns the response incrementally in chunks rather than all at once.
    -   `tools`: Provides callable function schemas the model can choose to use.
    -   `response_format`: Requests a specific output format (for example JSON) for reliable parsing.
-   Security:
    -   API keys should be treated like passwords
    -   store secrets in environment variables (not hardcoded)

### Quick example: call with common controls

``` python
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=messages,
    temperature=0.2,
    top_p=0.9,
    max_tokens=200,
    stream=False,
)
```

## Example response (as shown in lecture)

``` json
{
  "id": "6ATgaLyLFu6pkdUP8sG0sAo",
  "object": "chat.completion",
  "created": 1759511784,
  "model": "openai/gpt-oss-120b",
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "message": {
        "role": "assistant",
        "content": "Imagine a regular computer is like a light..."
      }
    }
  ],
  "usage_metadata": {
    "input_tokens": 27,
    "output_tokens": 622,
    "total_tokens": 649
  }
}
```

-   How to read this response:
    -   `choices` contains candidate outputs; by default, one output is returned.
    -   `choices[0].message.content` is the assistant text your app usually displays.
    -   `finish_reason` explains why generation stopped (for example `stop`).
    -   `model` confirms which model generated the answer.
    -   `usage_metadata` gives token counts used for cost and performance tracking.
    -   `id` and `created` are useful for logging, debugging, and trace correlation.

### Quick example: extract response fields

``` python
text = response.choices[0].message.content
tokens = response.usage_metadata["total_tokens"]
model_name = response.model
print(text, tokens, model_name)
```

## Memory and chat sequence

-   Key reality: models do not truly remember previous interactions.
-   Memory illusion comes from passing full conversation history in each request.
-   Message roles:
    -   `system`: behavior and constraints
    -   `user`: human/application input
    -   `assistant`: model output
-   Chat sequence pattern:
    -   system prompt -\> user message -\> assistant response -\> next user message -\> ...

### Example context window

``` json
[
  {
    "role": "system",
    "content": "You are a helpful Python tutor. Keep answers short."
  },
  {
    "role": "user",
    "content": "What is Python?"
  },
  {
    "role": "assistant",
    "content": "Python is a general-purpose programming language."
  },
  {
    "role": "user",
    "content": "How do I install it on Mac?"
  }
]
```

-   In this request, the full message list above is the context window input.
-   The model reads all messages together before producing the next response.
-   Token budget is shared across:
    -   message history (input tokens)
    -   new model reply (output tokens)
-   If history grows too large, older messages must be truncated or summarized.

### Quick example: append conversation history

``` python
history.append({"role": "user", "content": "What is a list in Python?"})
r1 = client.chat.completions.create(model=model, messages=history)
history.append({"role": "assistant", "content": r1.choices[0].message.content})
history.append({"role": "user", "content": "How do I add items to it?"})
```

## Tool calling

-   Tool calling lets the model return a function call (JSON arguments) instead of final text.
-   Tools are declared in context as function schemas.
-   Model decides when to call a tool based on query and tool descriptions.
-   Typical tool categories discussed:
    -   file operations
    -   document/internet search
    -   web interaction
    -   database access
    -   email/API integrations
    -   code execution

### Quick example: define and pass a tool

``` python
tools = [
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get weather for a city",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {"type": "string"},
          "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location"]
      }
    }
  }
]

resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "What's weather in Chicago today?"}],
    tools=tools,
    tool_choice="auto",
)
```

### Quick example: read tool call output

``` python
call = resp.choices[0].message.tool_calls[0].function
tool_name = call.name
tool_args_json = call.arguments
print(tool_name, tool_args_json)
```

## Pydantic and schema-first design

-   Why Pydantic was introduced:
    -   define typed classes clearly
    -   validate fields automatically
    -   serialize/deserialize JSON
    -   generate self-describing schemas
-   Pydantic `Field(...)` metadata helps the model by clarifying each property's intent.
-   Pydantic models can be nested for complex outputs.

### Quick example: Pydantic model

``` python
from pydantic import BaseModel, Field

class City(BaseModel):
    name: str = Field(description="City name")
    country: str = Field(description="Country name")
    population: int = Field(description="Population as integer")
```

## Structured outputs

-   Free text is good for humans, but brittle for application logic.
-   Structured outputs constrain model responses to schema shape (JSON-backed).
-   Benefits:
    -   predictable parsing
    -   validation guardrails
    -   easier handoff between system components/agents
-   Class example discussed: city details with fixed fields like `name`, `country`, `population`.

### Quick example: request JSON output

``` python
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Provide details about Paris."}],
    response_format={"type": "json_object"},
    temperature=0,
)
```

### Quick example: validate output with Pydantic

``` python
import json
city = City.model_validate(json.loads(response.choices[0].message.content))
print(city.model_dump())
```

## Streaming and observability

-   Streaming concept:
    -   traditional mode waits for full response
    -   streaming returns chunks/tokens in real time
    -   better perceived latency and progressive UX
-   Observability focus:
    -   tracing captures exact prompt/response flow
    -   useful for debugging and replay
    -   traces can later feed eval datasets
-   Tool shown in lecture: LangSmith tracing for OpenAI SDK calls.

### Quick example: streaming response

``` python
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Give me 3 Python tips."}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="")
```

### Quick example: LangSmith tracing wrapper

``` python
from openai import OpenAI
from langsmith.wrappers import wrap_openai

client = wrap_openai(OpenAI(api_key=api_key, base_url=base_url))
```

## Key terms

-   `Stateless protocol`: no server-side conversation memory by default.
-   `Context window`: all tokens the model can consider in a call.
-   `Tool schema`: machine-readable function description exposed to the model.
-   `Structured output`: response constrained to a declared schema.
-   `Tracing`: record of calls, prompts, outputs, and sequence.

## 1 diagram

``` mermaid
flowchart LR
  A["Build request context"] --> B["chat.completions.create"]
  B --> C{"Model output type"}
  C -->|Text| D["Use message.content"]
  C -->|Tool call| E["Parse tool name + JSON args"]
  E --> F["Execute tool in app"]
  F --> G["Append result into messages"]
  G --> B
```

## 1 mini experiment

-   Run the same user query in two modes:
    -   Mode A: normal text response
    -   Mode B: structured JSON response mapped to a Pydantic model
-   Compare:
    -   parsing reliability
    -   downstream coding effort
    -   error handling complexity

## 1 question to research

-   Which observability metrics are most useful for agent reliability: tool success rate, schema-valid response rate, recovery rate, or latency per successful task?

## References from lecture

-   `2. Interacting with Model API - v3.pdf`
-   OpenAI Python SDK docs
-   Pydantic docs
-   LangSmith docs