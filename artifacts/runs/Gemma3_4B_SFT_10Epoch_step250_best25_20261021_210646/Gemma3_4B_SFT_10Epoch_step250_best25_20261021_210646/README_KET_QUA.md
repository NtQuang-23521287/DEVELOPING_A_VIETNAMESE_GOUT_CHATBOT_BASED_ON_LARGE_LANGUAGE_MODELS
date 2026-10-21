# Gemma3-4B — SFT QLoRA 10 Epoch

- Completed: **250 optimizer steps**
- Best checkpoint: **checkpoint-25** (Epoch 1)
- Best eval_loss: **2.080469131**
- Train loss summary: **None**
- Dataset: **197 train valid / 22 validation** (the training log).
- Precision note: Unsloth switched Gemma3 to float32 on T4.
- The clinical training answers are **DRAFT**, not medically approved.
- This backup has **LoRA weights**, not full Google Gemma3 base model.

## Eval Loss per Epoch

| Epoch | Step | Eval loss |
|---:|---:|---:|
| 1 | 25 | 2.080469131 |
| 2 | 50 | 2.467909575 |
| 3 | 75 | 2.487312555 |
| 4 | 100 | 2.572119713 |
| 5 | 125 | 2.630132198 |
| 6 | 150 | 2.662732601 |
| 7 | 175 | 2.617132425 |
| 8 | 200 | 2.587229013 |
| 9 | 225 | 2.619153261 |
| 10 | 250 | 2.616969824 |

## Khôi phục

- Best adapter/checkpoint: `run_artifacts/checkpoint-25/`
- Root adapter: `run_artifacts/` (verify matching best checkpoint separately).
- `experiment_snapshot/` contains the available source YAML, JSONL, Python scripts.
- `sha256_manifest.json` and `BACKUP_COMPLETE.json` verify archive integrity.
- To infer: load the compatible `google/gemma-3-4b-it` base model and PEFT adapter.
