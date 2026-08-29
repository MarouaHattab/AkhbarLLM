# news-finetuning

Arabic news pipeline: distill labels, format and train a Qwen LoRA adapter,
evaluate, serve locally with vLLM, then load-test.

Python 3.12. Commands below use [uv](https://docs.astral.sh/uv/) from the
repository root. Put `OPENAI_API_KEY`, `GEMINI_API_KEY`, and `HF_TOKEN` in
`src/.env` as needed.

## Distill

```powershell
uv run --group distillation python -m src.workflows.generate_distillation_dataset
```

## Format, register, prepare, train

```powershell
uv run --group train python -m src.workflows.format_finetuning_dataset
uv run --group train python -m src.workflows.register_llamafactory_datasets
uv run --group train python -m src.workflows.prepare_finetuning
```

`prepare_finetuning` copies the training YAML and prints the LLaMA-Factory
command. It does not start training. The adapter is written to
`outputs/models/news-finetune`.

## Evaluate and benchmark

```powershell
uv run --group evaluation python -m src.workflows.evaluate_models --model all
uv run --group evaluation python -m src.workflows.benchmark_models
```

`--model all` evaluates `qwen`, `finetuned`, `gemini`, and `openai`. Served
vLLM is a separate command (`infer_vllm`) after the adapter is up.

## Serve

vLLM runs in WSL (Linux), not Docker. From Windows:

```powershell
.\deployment\vllm\serve.ps1
uv run --group serve python -m src.workflows.check_vllm --wait
uv run --group serve python -m src.workflows.infer_vllm
```

`check_vllm` and `infer_vllm` use `load_language_model("vllm")` against
`http://localhost:8000/v1`.

## Locust (same idea as the Kaggle notebook)

```powershell
uv run --group serve locust `
  -f tests/load/locustfile.py `
  --headless `
  --host=http://localhost:8000 `
  -u 20 `
  -r 1 `
  -t 60s `
  --html=outputs/LoadTesting/locust-results.html
```

This posts Arabic `Faker("ar_SA")` prompts to `POST /v1/completions` with
`model=news-lora`, `max_tokens=512`, and `temperature=0.3`. Successful
`prompt` / `response` pairs are appended to
`outputs/LoadTesting/vllm-tokens.jsonl`. Open the HTML file in Chrome or Edge.

## Count tokens

```powershell
uv run --group serve python -m src.workflows.analyze_vllm_load
```

This loads the JSONL and counts tokens with
`AutoTokenizer` for `Qwen/Qwen2.5-1.5B-Instruct`, then prints:

```
Loaded N responses.
Total Input Tokens: ...
Total Output Tokens: ...
```

## Tests

```powershell
uv run --group evaluation --group serve --group test pytest tests/unit -q
```
