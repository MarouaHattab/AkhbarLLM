# News model fine-tuning

This project evaluates `Qwen/Qwen2.5-1.5B-Instruct` before fine-tuning and
compares it with `gemini-3.1-flash-lite` and `gpt-4o-mini`. All providers use
the same extraction and translation prompts and the same Pydantic validation.
Gemini or OpenAI can later be used as a teacher for knowledge distillation.

## Environment

Copy `src/.env.example` to `src/.env` and fill only the keys you need:

```dotenv
HF_TOKEN=
GEMINI_API_KEY=
OPENAI_API_KEY=
WANDB_API_KEY=
```

`src/.env` is ignored by Git. Model IDs and generation parameters belong in
`src/helpers/config.py`, not in the environment file.

Install the evaluation dependencies:

```powershell
uv sync --group evaluation
```

## Pre-fine-tuning evaluation

Run both extraction and translation with one provider:

```powershell
uv run python -m src.workflows.evaluate_models --model qwen --task both
uv run python -m src.workflows.evaluate_models --model gemini --task both
uv run python -m src.workflows.evaluate_models --model openai --task both
```

Compare all providers using the same story and prompts:

```powershell
uv run python -m src.workflows.evaluate_models --model all --task both
```

The `all` order is Qwen, Gemini, then OpenAI. You can choose only one task with
`--task extraction` or `--task translation`. The default input is
`data/examples/story.txt`; override it with `--story` when needed.

Gemini and OpenAI commands send the story to external APIs and may incur
charges. The automated tests inject fake clients and never call those APIs.

## Source structure

- `src/controllers`: provider-independent evaluation and schema validation
- `src/helpers`: configuration, environment-key loading, and authentication
- `src/models`: shared interface and Qwen, Gemini, and OpenAI adapters
- `src/tasks`: extraction and translation prompt builders
- `src/templates`: reusable system prompts
- `src/utils`: input-loading utilities
