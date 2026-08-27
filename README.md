## News Fine-Tuning

### Prepare fine-tuning

Set these API keys in `src/.env`:

```dotenv
HF_TOKEN=your_hugging_face_token
WANDB_API_KEY=your_wandb_api_key
```

Clone LLaMA-Factory into `LLaMA-Factory`, format and register the datasets,
then run the preparation command:

```powershell
uv run --group train python -m src.workflows.prepare_finetuning
```

This command authenticates Hugging Face and W&B, registers the datasets,
copies `news_finetune.yaml` into LLaMA-Factory, and creates
`outputs/models/news-finetune`. It does not start training.

For Kaggle or another checkout, override the portable paths:

```bash
python -m src.workflows.prepare_finetuning \
  --llamafactory-dir /kaggle/working/LLaMA-Factory \
  --train-path /kaggle/working/llamafactory-finetune-data/train.json \
  --val-path /kaggle/working/llamafactory-finetune-data/val.json
```

Start training manually later from the LLaMA-Factory directory:

```bash
llamafactory-cli train examples/train_lora/news_finetune.yaml
```

The LoRA checkpoints are stored under `outputs/models/news-finetune` and are
ignored by Git. The training configuration privately pushes checkpoints to
`marouaht/news-analyzer` on the Hugging Face Hub.
