# Lab: Groq + OpenAI SDK Hands-on

## Objective
Run end-to-end examples for:
- Basic model call
- Multi-turn chat history ("memory")
- Tool calling
- Structured output with Pydantic validation
- Optional LangSmith tracing

## Setup
1. Create and activate a virtual environment (recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Copy environment file and add your key:
```bash
cp .env.example .env
```
Then set:
- `GROQ_API_KEY=...`

Optional tracing setup:
- `LANGSMITH_TRACING=true`
- `LANGSMITH_API_KEY=...`
- `LANGSMITH_ENDPOINT=https://api.smith.langchain.com`
- `LANGSMITH_PROJECT=lab-00-model-api-groq`

## Run
From this folder:
```bash
python 01_basic_chat.py
python 02_multi_turn_memory.py
python 03_tool_calling.py
python 04_structured_output.py
python 05_langsmith_tracing_smoke.py
python 06_structured_math_reasoning.py
```

## Verify LangSmith tracing
1. In `.env`, set:
- `LANGSMITH_TRACING=true`
- `LANGSMITH_API_KEY=...`
- `LANGSMITH_ENDPOINT=https://api.smith.langchain.com`
- `LANGSMITH_PROJECT=lab-00-model-api-groq`
2. Run:
```bash
python 05_langsmith_tracing_smoke.py
```
3. Check your LangSmith project for a new trace entry.

## Notes
- Default model is `openai/gpt-oss-120b`.
- You can override model with:
```bash
export GROQ_MODEL="openai/gpt-oss-120b"
```
- This lab uses OpenAI-compatible endpoints via:
`https://api.groq.com/openai/v1`
