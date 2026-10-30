# SFT training comparison

The per-model Training Loss is the arithmetic mean of recorded step losses.
Validation Loss is measured on the validation split.
Different model architectures/tokenizers mean absolute loss values are **not** a direct measure of which chatbot is medically better.
Choose candidate checkpoints on validation, and evaluate actual responses (Judge v2 / safety) before selecting a final model.

## Qwen3-8B (LR=1e-4, 10ep)
- Best epoch: 6.0
- Best validation loss: 1.545760
- Final validation loss: 1.625390
- Final versus best: +5.15%

## Gemma3-4B (LR=2e-5, 10ep)
- Best epoch: 5.0
- Best validation loss: 2.048518
- Final validation loss: 2.139963
- Final versus best: +4.46%
