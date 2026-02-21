"""LoRA fine-tuning of Qwen2.5-7B for SCOTUS vote prediction."""

import glob
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-7B"
# MAX_SEQ_LENGTH = 32768
MAX_SEQ_LENGTH = 16384

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.05

DATA_DIR = "data/transcripts"
OUTPUT_DIR = "output/scotus-lora"

VOTES_TAG = "<votes>"


# ── Data loading ──────────────────────────────────────────────────────────────


def load_transcripts(data_dir: str) -> list[dict]:
    """Load transcript files and split into prompt/completion pairs.

    Prompt = everything up to and including <votes>
    Completion = everything after <votes>
    """
    samples = []
    paths = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    for path in paths:
        with open(path) as f:
            text = f.read()

        if VOTES_TAG not in text:
            print(f"Warning: skipping {path} — no {VOTES_TAG} tag found")
            continue

        idx = text.index(VOTES_TAG) + len(VOTES_TAG)
        prompt = text[:idx]
        completion = text[idx:].strip()

        if not completion:
            print(f"Warning: skipping {path} — empty completion")
            continue

        samples.append({"prompt": prompt, "completion": completion})

    print(f"Loaded {len(samples)} samples from {data_dir}")
    return samples


# ── Tokenization ──────────────────────────────────────────────────────────────


def tokenize_sample(
    sample: dict, tokenizer, max_length: int
) -> dict:
    """Tokenize a single sample with left-truncation of the prompt.

    The completion (+ EOS) is never truncated. If the total exceeds max_length,
    the prompt is left-truncated so the end of the transcript (nearest to the
    votes) is preserved.

    Labels: -100 for prompt tokens, real IDs for completion tokens.
    """
    completion_ids = tokenizer.encode(
        sample["completion"], add_special_tokens=False
    ) + [tokenizer.eos_token_id]

    prompt_ids = tokenizer.encode(sample["prompt"], add_special_tokens=False)

    max_prompt_len = max_length - len(completion_ids)
    if max_prompt_len <= 0:
        raise ValueError(
            f"Completion ({len(completion_ids)} tokens) exceeds max_length "
            f"({max_length}). Increase MAX_SEQ_LENGTH."
        )

    # Left-truncate: keep the rightmost tokens of the prompt
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def prepare_dataset(
    samples: list[dict], tokenizer, max_length: int
) -> tuple[Dataset, Dataset]:
    """Tokenize all samples and return train/eval datasets (90/10 split)."""
    tokenized = [tokenize_sample(s, tokenizer, max_length) for s in samples]
    dataset = Dataset.from_list(tokenized)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    print(
        f"Split: {len(split['train'])} train, {len(split['test'])} eval samples"
    )
    return split["train"], split["test"]


# ── Model setup ───────────────────────────────────────────────────────────────


def setup_model_and_tokenizer():
    """Load Qwen2.5-7B in bf16 with flash attention and apply LoRA."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ── Training ──────────────────────────────────────────────────────────────────


def train():
    """Main training entry point."""
    model, tokenizer = setup_model_and_tokenizer()
    samples = load_transcripts(DATA_DIR)
    train_dataset, eval_dataset = prepare_dataset(samples, tokenizer, MAX_SEQ_LENGTH)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        bf16=True,
        gradient_checkpointing=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=2,
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # Save final adapter
    final_path = os.path.join(OUTPUT_DIR, "final_adapter")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Saved adapter to {final_path}")


if __name__ == "__main__":
    train()
