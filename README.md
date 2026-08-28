## Evaluate the fine-tuned model

The `finetuned` provider loads `Qwen/Qwen2.5-1.5B-Instruct` and
attaches the local LoRA adapter from:

```text
outputs/models/news-finetune
```

Evaluate extraction and translation with the default story in
`data/examples/story.txt`:

```powershell
uv run --group evaluation python -m src.workflows.evaluate_models --model finetuned --task both
```

Save the report directly from Python so Arabic text stays readable on
Windows:

```powershell
uv run --group evaluation python -m src.workflows.evaluate_models --model finetuned --task both --output outputs/Evaluation/qwen-finetuned-evaluation.txt
```

Avoid piping evaluation output through PowerShell and then writing it
with `Set-Content`. On Windows, that capture step can misread UTF-8 as
the legacy CP850 code page and turn Arabic into garbled text such as
`Ï»┘êÏ▒`.

To compare the base Qwen model, fine-tuned Qwen model, Gemini, and OpenAI:

```powershell
uv run --group evaluation python -m src.workflows.evaluate_models --model all --task both
```

The `finetuned` command runs the local model and does not call a paid API.
The `all` command also calls Gemini and OpenAI and may incur API charges.
