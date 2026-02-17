# Lab 01: Prompt Engineering Tutorial (Week 3)

## Objective
Run the Week 3 prompt engineering practical end-to-end:
- Basic prompting (zero/single/few-shot)
- CRAFT + instruction prompting
- Reasoning prompts (CoT, ToT, DoT)
- Prompt chaining
- Workflow with tools
- ReAct loop over multiple files

## Setup
1. Create and activate virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create `.env`:
```bash
cp .env.example .env
```
Then set `GROQ_API_KEY`.

## Step-by-step run order
From this folder (`labs/lab-01-prompting`):
```bash
python 01_basic_prompting.py
python 02_craft_instruction.py
python 03_reasoning_cot_tot_dot.py
python 04_prompt_chaining.py
python 05_workflow_with_tools.py
python 06_react_loop.py
```

## What each script demonstrates
- `01_basic_prompting.py`: zero-shot vs single-shot vs few-shot behavior.
- `02_craft_instruction.py`: structured CRAFT prompt and process-oriented instruction prompt.
- `03_reasoning_cot_tot_dot.py`: CoT, ToT, and DoT prompt patterns.
- `04_prompt_chaining.py`: multi-step decomposition where output of one step feeds the next.
- `05_workflow_with_tools.py`: fixed workflow that calls a file-reading tool then summarizes.
- `06_react_loop.py`: iterative reason+act loop using `ListFilesTool` and `ReadFileTool`.

## Exercise mapping (from tutorial)
- Exercise 1 (Technique comparison): run `01` and `03`, compare output quality.
- Exercise 2 (Progressive refinement): modify one task from zero-shot -> few-shot -> CoT/ToT in `01` + `03`.
- Exercise 3 (ReAct): extend `06_react_loop.py` with extra tool constraints or scoring.

## Notes
- Default model is `openai/gpt-oss-120b`; override with `GROQ_MODEL` in `.env`.
- You can test smaller models too (if available) to compare prompt sensitivity.
