# News model fine-tuning

This project compares the base `Qwen/Qwen2.5-1.5B-Instruct` model with Gemini
and OpenAI before fine-tuning, generates teacher-labelled extraction data, and
keeps prompts, schemas, provider adapters, and workflows separated under
`src/`.

## Environment

Copy `src/.env.example` to `src/.env` and fill only the keys you use:

```dotenv
HF_TOKEN=
GEMINI_API_KEY=
OPENAI_API_KEY=
WANDB_API_KEY=
```

Never commit `src/.env`.

## Pre-fine-tuning evaluation

```powershell
uv sync --group evaluation
uv run --group evaluation python -m src.workflows.evaluate_models --model qwen --task both
uv run --group evaluation python -m src.workflows.evaluate_models --model gemini --task both
uv run --group evaluation python -m src.workflows.evaluate_models --model openai --task both
uv run --group evaluation python -m src.workflows.evaluate_models --model all --task both
```

## o4-mini knowledge distillation

The distillation workflow follows the Kaggle extraction loop:

- reads all 2,400 records from `data/raw/news-sample.jsonl`;
- shuffles them with seed `101`;
- overwrites `data/datasets/sft.jsonl` at the start;
- calls `o4-mini` sequentially for each non-empty story;
- validates each JSON response with `NewsDetails`;
- continues after API, empty-response, JSON, or schema failures;
- reports progress every 10 successful samples;
- reports actual token usage and estimated cost at the end.

Install only the distillation dependencies:

```powershell
uv sync --group distillation
```

The following command immediately starts paid API calls for the entire input
file and replaces any existing `data/datasets/sft.jsonl`:

```powershell
uv run --group distillation python -m src.workflows.generate_distillation_dataset
```

The configured standard `o4-mini` prices per 1 million tokens are:

- uncached input: `$1.10`
- cached input: `$0.275`
- output, including billed reasoning tokens: `$4.40`

`o4-mini` is deprecated/succeeded by a newer model, but this project preserves
the explicitly selected teacher model. Confirm that your OpenAI project still
has access before starting the full run.

## Source structure

- `src/controllers`: evaluation and response validation
- `src/helpers`: configuration, secrets, and authentication
- `src/models`: schemas and Qwen/Gemini/OpenAI adapters
- `src/tasks`: task-specific prompt builders
- `src/templates`: reusable system prompts
- `src/utils`: story and JSONL utilities
- `src/workflows`: evaluation and distillation orchestration
