Local vLLM serving and Locust load testing for the `news-lora` adapter.

## Serve

```powershell
.\deployment\vllm\serve.ps1
uv run --group serve python -m src.workflows.check_vllm --wait
```

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
