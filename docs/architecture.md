# Architecture v1

```text
Raw Sources ──> Validation/Manifest ──> KB Build ──> FAISS + Metadata
                                      │
Candidate Testset ──> Validation ─────┼────> Evaluation Runner
                                      │          │
Model Config ──> HF Backend ──> ChatEngine <────┘
                              │   │
                              │   ├── history (case-local only)
                              │   └── RAG contexts
                              ▼
                         predictions.jsonl
                              │
                  ┌───────────┴───────────┐
                  ▼                       ▼
             LLM Judge                 RAGAS
                  └───────────┬───────────┘
                              ▼
                          MLflow/report
```

## Boundaries

`reference` không tồn tại trong chữ ký `ChatEngine.chat`. Reference chỉ được phép vào evaluator sau khi answer đã được sinh. Đây là guardrail chống test leakage.

Baseline/SFT/DPO phải implement cùng interface model backend để Evaluation Runner không cần viết lại.
