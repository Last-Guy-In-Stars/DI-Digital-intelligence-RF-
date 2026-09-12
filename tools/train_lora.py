#!/usr/bin/env python3
"""
LoRA v3.1: надёжная конвертация GGUF с fallback.
"""
import json
import os
import shutil
import glob
import subprocess

MODEL_NAME = "unsloth/Qwen3-8B-bnb-4bit"
DATASET_PATH = "tools/lora_dataset.jsonl"
OUTPUT_DIR = "tools/lora_output"
MERGED_GGUF = "brain/models/qwen3-8b-leta-lora.gguf"
EPOCHS = 200
LR = 5e-5


def clean_disk():
    """Освобождает место перед конвертацией GGUF."""
    print("\n--- Очистка диска ---")
    # Удалить кэш huggingface (слияние уже завершено)
    cache = os.path.expanduser("~/.cache/huggingface")
    if os.path.exists(cache):
        shutil.rmtree(cache)
        print(f"удалён кэш: {cache}")
    # Удалить старые checkpoints
    for d in glob.glob(f"{OUTPUT_DIR}/checkpoint-*"):
        shutil.rmtree(d)
        print(f"удалён checkpoint: {d}")
    # Удалить adapters (не нужны после слияния)
    adapters = f"{OUTPUT_DIR}/adapters"
    if os.path.exists(adapters):
        shutil.rmtree(adapters)
        print(f"удалены adapters")
    # Проверить свободное место
    r = subprocess.run(["df", "-h", "/"], capture_output=True, text=True)
    print(r.stdout.splitlines()[-1])


def convert_gguf_unsloth(model, tokenizer):
    """Попытка конвертации через unsloth."""
    print("\n--- Конвертация GGUF (unsloth) ---")
    try:
        os.environ["UNSLOTH_DISK_PREFLIGHT"] = "0"
        model.save_pretrained_gguf(
            f"{OUTPUT_DIR}/gguf", tokenizer, quantization_method="q4_k_m"
        )
        files = glob.glob(f"{OUTPUT_DIR}/gguf/*.gguf")
        if files:
            return files[0]
    except Exception as e:
        print(f"unsloth GGUF не сработал: {e}")
    return None


def convert_gguf_llamacpp():
    """Fallback: конвертация через llama.cpp convert script."""
    print("\n--- Конвертация GGUF (llama.cpp) ---")
    merged_dir = f"{OUTPUT_DIR}/merged"
    if not os.path.exists(merged_dir):
        print(f"merged не найден: {merged_dir}")
        return None

    # Метод 1: llama.cpp через pip
    try:
        script = os.path.join(
            os.path.dirname(__import__("llama_cpp").__file__), "convert_hf_to_gguf.py"
        )
        if os.path.exists(script):
            result = subprocess.run(
                [
                    "python3", script,
                    merged_dir,
                    "--outfile", MERGED_GGUF,
                    "--outtype", "f16",
                ],
                capture_output=True, text=True, timeout=600,
            )
            if os.path.exists(MERGED_GGUF):
                print(f"GGUF (f16): {MERGED_GGUF}")
                return MERGED_GGUF
            print(f"ошибка: {result.stderr[-200:]}")
    except Exception as e:
        print(f"метод 1 не сработал: {e}")

    # Метод 2: llama.cpp convert.py из установленного бинарника
    try:
        # Найти llama.cpp установленный unsloth'ом
        unsloth_llamacpp = os.path.expanduser("~/.unsloth/llama.cpp")
        if os.path.exists(unsloth_llamacpp):
            convert = f"{unsloth_llamacpp}/convert_hf_to_gguf.py"
            if os.path.exists(convert):
                result = subprocess.run(
                    [
                        "python3", convert,
                        merged_dir,
                        "--outfile", MERGED_GGUF,
                        "--outtype", "f16",
                    ],
                    capture_output=True, text=True, timeout=600,
                )
                if os.path.exists(MERGED_GGUF):
                    print(f"GGUF (f16): {MERGED_GGUF}")
                    return MERGED_GGUF
                print(f"ошибка: {result.stderr[-200:]}")
    except Exception as e:
        print(f"метод 2 не сработал: {e}")

    # Метод 3: просто скопировать GGUF если unsloth его куда-то положил
    for pattern in [
        f"{OUTPUT_DIR}/**/*.gguf",
        f"brain/models_gguf/*.gguf",
        os.path.expanduser("~/.unsloth/**/*.gguf"),
    ]:
        for f in glob.glob(pattern, recursive=True):
            if os.path.getsize(f) > 3e9:  # > 3GB = модель
                shutil.copy(f, MERGED_GGUF)
                print(f"скопирован: {f} → {MERGED_GGUF}")
                return MERGED_GGUF

    return None


def main():
    from unsloth import FastLanguageModel
    import torch

    print("=== LETA LoRA v3.1 ===")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    print(f"Примеров: {len(data)}, эпох: {EPOCHS}")

    from datasets import Dataset
    texts = []
    for item in data:
        messages = item["messages"]
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        texts.append("\n".join(parts))
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=2048, padding=False)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    from transformers import TrainingArguments
    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized,
        dataset_text_field="input_ids",
        max_seq_length=2048,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            warmup_ratio=0.05,
            learning_rate=LR,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=20,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            save_strategy="no",  # не сохранять checkpoints (экономим место)
            report_to="none",
        ),
    )

    print("\nОбучение...")
    trainer.train()

    print("\n--- Слияние ---")
    model.save_pretrained_merged(
        f"{OUTPUT_DIR}/merged", tokenizer, save_method="merged_16bit",
    )
    print(f"merged сохранён: {OUTPUT_DIR}/merged")

    # Освободить место
    clean_disk()

    # Попытка конвертации GGUF (несколько методов)
    gguf = convert_gguf_unsloth(model, tokenizer)
    if not gguf:
        gguf = convert_gguf_llamacpp()

    if gguf:
        if gguf != MERGED_GGUF:
            shutil.copy(gguf, MERGED_GGUF)
        size_gb = os.path.getsize(MERGED_GGUF) / 2**30
        print(f"\n{'='*50}")
        print(f"ГОТОВО! {MERGED_GGUF} ({size_gb:.1f} GB)")
        print(f"{'='*50}")
    else:
        print(f"\n{'!'*50}")
        print(f"GGUF не создан автоматически!")
        print(f"merged модель в: {OUTPUT_DIR}/merged")
        print(f"Конвертируй вручную:")
        print(f"  cd {OUTPUT_DIR}/merged")
        print(f"  python3 ~/.unsloth/llama.cpp/convert_hf_to_gguf.py . --outfile {MERGED_GGUF} --outtype f16")
        print(f"{'!'*50}")


if __name__ == "__main__":
    main()
