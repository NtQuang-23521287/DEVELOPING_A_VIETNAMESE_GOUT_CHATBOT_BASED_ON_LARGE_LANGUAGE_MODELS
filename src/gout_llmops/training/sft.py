from __future__ import annotations
import json
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

def train_sft(config: dict) -> None:
    print("🚀 Bắt đầu Phase 3: Supervised Fine-Tuning...")

    # Đọc đường dẫn từ file config YAML
    jsonl_path = config.get("training_data", "data/sft/gout_sft_dataset.jsonl")
    output_dir = config.get("output_dir", "artifacts/runs/sft_v1")
    
    # 1. Cấu hình Model & LoRA
    max_seq_length = 2048
    model_name = "unsloth/Qwen-8B-Instruct-bnb-4bit"
    
    print(f"Đang tải mô hình: {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )

    # Lấy thông số QLoRA từ config
    qlora_cfg = config.get("qlora", {})
    model = FastLanguageModel.get_peft_model(
        model,
        r = qlora_cfg.get("lora_r", 16),
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = qlora_cfg.get("lora_alpha", 32),
        lora_dropout = qlora_cfg.get("lora_dropout", 0), 
        bias = "none",
        use_gradient_checkpointing = "unsloth",
    )

    alpaca_prompt = "{instruction}\n### Câu hỏi:\n{input}\n### Trả lời:\n{output}"

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for inst, inp, out in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction=inst, input=inp, output=out) + tokenizer.eos_token
            texts.append(text)
        return { "text" : texts }

    print(f"Đang đọc dữ liệu từ {jsonl_path}...")
    dataset = load_dataset("json", data_files=jsonl_path, split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True)

    # 3. Tiến hành huấn luyện
    print("Bắt đầu chạy Trainer...")
    trainer_cfg = config.get("trainer", {})
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        args = TrainingArguments(
            per_device_train_batch_size = trainer_cfg.get("per_device_train_batch_size", 1),
            gradient_accumulation_steps = trainer_cfg.get("gradient_accumulation_steps", 8),
            warmup_steps = 5,
            num_train_epochs = trainer_cfg.get("num_train_epochs", 2),
            learning_rate = trainer_cfg.get("learning_rate", 2e-4),
            fp16 = not FastLanguageModel.is_bfloat16_supported(),
            bf16 = trainer_cfg.get("bf16", True) and FastLanguageModel.is_bfloat16_supported(),
            logging_steps = 1, # Set = 1 để terminal liên tục báo cáo tiến độ
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = config.get("seed", 2026),
            output_dir = output_dir,
        ),
    )
    
    trainer.train()
    
    # Lưu trọng số mô hình
    print(f"Lưu trọng số LoRA vào: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("🎉 Hoàn thành SFT Phase 3!")