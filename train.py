"""QLoRA fine-tuning of Qwen2.5 for SCOTUS vote prediction."""

import argparse
import glob
import os
import re

import torch
from tqdm import tqdm
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ── Config ────────────────────────────────────────────────────────────────────

MODELS = {
    "7b": "Qwen/Qwen2.5-7B",
    "3b": "Qwen/Qwen2.5-3B",
    "1.5b": "Qwen/Qwen2.5-1.5B",
    "0.5b": "Qwen/Qwen2.5-0.5B",
}
DEFAULT_MODEL = "7b"

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
NUM_EPOCHS = 1
WARMUP_RATIO = 0.05

DATA_DIR = "case_transcripts"
MAX_TOKENS_DISCARD = 15000
OUTPUT_DIR = "output/scotus-lora"

VOTES_DELIMITER = "\n---\nJUSTICE VOTES:\n"
EVAL_YEAR = "2019"


# ── Data loading ──────────────────────────────────────────────────────────────


def load_transcripts(data_dir: str) -> list[dict]:
    """Load transcript files and split into prompt/completion pairs.

    Prompt = everything up to and including the JUSTICE VOTES: header
    Completion = the vote lines and outcome line
    """
    samples = []
    paths = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    for path in paths:
        with open(path) as f:
            text = f.read()

        if VOTES_DELIMITER not in text:
            print(f"Warning: skipping {path} — no JUSTICE VOTES delimiter found")
            continue

        idx = text.index(VOTES_DELIMITER) + len(VOTES_DELIMITER)
        prompt = text[:idx]
        completion = text[idx:].strip()

        if not completion:
            print(f"Warning: skipping {path} — empty completion")
            continue

        filename = os.path.basename(path)
        samples.append({
            "prompt": prompt,
            "completion": completion,
            "filename": filename,
        })

    print(f"Loaded {len(samples)} samples from {data_dir}")
    return samples


def split_by_year(
    samples: list[dict], eval_year: str
) -> tuple[list[dict], list[dict]]:
    """Split samples into train/eval based on filename year prefix."""
    train_samples = []
    eval_samples = []
    for s in samples:
        if s["filename"].startswith(eval_year):
            eval_samples.append(s)
        else:
            train_samples.append(s)
    print(f"Year split: {len(train_samples)} train, "
          f"{len(eval_samples)} eval (year={eval_year})")
    return train_samples, eval_samples


# ── Tokenization ──────────────────────────────────────────────────────────────


TOKENIZE_BATCH_SIZE = 256


def _batch_encode(texts: list[str], tokenizer, batch_size: int) -> list[list[int]]:
    """Tokenize texts in batches for speed."""
    all_ids = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(batch, add_special_tokens=False,
                            return_attention_mask=False)
        all_ids.extend(encoded["input_ids"])
    return all_ids


def tokenize_and_filter(
    samples: list[dict], tokenizer, max_length: int, label: str
) -> list[dict]:
    """Tokenize samples in batches, print lengths, discard those exceeding MAX_TOKENS_DISCARD."""
    # Batch-tokenize prompts and completions separately
    prompts = [s["prompt"] for s in samples]
    completions = [s["completion"] for s in samples]

    print(f"  Batch-tokenizing {len(samples)} prompts ...")
    prompt_ids_list = _batch_encode(prompts, tokenizer, TOKENIZE_BATCH_SIZE)

    print(f"  Batch-tokenizing {len(samples)} completions ...")
    completion_ids_list = _batch_encode(completions, tokenizer, TOKENIZE_BATCH_SIZE)

    # First subtokens of vote words for vote_mask
    pet_first = tokenizer.encode("Petitioner", add_special_tokens=False)[0]
    res_first = tokenizer.encode("Respondent", add_special_tokens=False)[0]
    vote_tokens = {pet_first, res_first}

    # Assemble samples with left-truncation and label masking
    tokenized = []
    for prompt_ids, completion_ids in tqdm(
        zip(prompt_ids_list, completion_ids_list),
        total=len(samples),
        desc=f"Assembling {label}",
    ):
        completion_ids = completion_ids + [tokenizer.eos_token_id]

        max_prompt_len = max_length - len(completion_ids)
        if max_prompt_len <= 0:
            continue  # completion alone exceeds max_length, skip

        # Left-truncate: keep the rightmost tokens of the prompt
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        attention_mask = [1] * len(input_ids)
        vote_mask = [1 if (lb in vote_tokens) else 0 for lb in labels]

        tokenized.append({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "vote_mask": vote_mask,
        })

    lengths = [len(t["input_ids"]) for t in tokenized]
    print(f"\n── Token lengths: {label} ({len(lengths)} samples) ──")
    print(f"  Max: {max(lengths):,}  Min: {min(lengths):,}  "
          f"Mean: {sum(lengths)//len(lengths):,}")

    before = len(tokenized)
    tokenized = [t for t in tokenized if len(t["input_ids"]) <= MAX_TOKENS_DISCARD]
    if len(tokenized) < before:
        print(f"  Discarded {before - len(tokenized)} samples exceeding "
              f"{MAX_TOKENS_DISCARD:,} tokens")
    print()
    return tokenized


# ── Vote parsing & scoring ────────────────────────────────────────────────────


def parse_votes(text: str) -> dict[str, str]:
    """Parse vote lines into {justice_name: side} dict.

    Expects lines like: "John G. Roberts, Jr.: Petitioner"
    Normalizes names to lowercase for comparison.
    """
    votes = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("OUTCOME:"):
            continue
        # Split on last ": " to handle names with commas
        if ": " not in line:
            continue
        name, side = line.rsplit(": ", 1)
        side = side.strip().lower()
        if side in ("petitioner", "respondent"):
            votes[name.strip().lower()] = side
    return votes


def case_result(votes: dict[str, str]) -> str | None:
    """Determine case winner by majority vote."""
    if not votes:
        return None
    pet = sum(1 for v in votes.values() if v == "petitioner")
    resp = sum(1 for v in votes.values() if v == "respondent")
    if pet > resp:
        return "petitioner"
    elif resp > pet:
        return "respondent"
    return None


def score_predictions(
    predicted_text: str, ground_truth_text: str
) -> dict:
    """Compare predicted votes against ground truth.

    Returns per-justice correctness and case result correctness.
    Order-agnostic: matches by justice name.
    """
    pred_votes = parse_votes(predicted_text)
    true_votes = parse_votes(ground_truth_text)

    justice_results = {}
    for justice, true_side in true_votes.items():
        if justice in pred_votes:
            justice_results[justice] = pred_votes[justice] == true_side
        else:
            justice_results[justice] = None  # missing

    pred_result = case_result(pred_votes)
    true_result = case_result(true_votes)
    case_correct = pred_result == true_result if true_result else None

    return {
        "justice_results": justice_results,
        "case_correct": case_correct,
        "pred_votes": pred_votes,
        "true_votes": true_votes,
    }


# ── Model setup ───────────────────────────────────────────────────────────────


def setup_model_and_tokenizer(model_name: str):
    """Load model in 4-bit (QLoRA) with SDPA and apply LoRA."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        attn_implementation="sdpa",
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


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate_generation(model, tokenizer, eval_samples: list[dict]):
    """Generate vote predictions for eval samples and score them."""
    model.eval()
    model.config.use_cache = True

    all_justice_results = {}  # justice -> list of bool
    case_results = []

    print(f"\n── Generating predictions for {len(eval_samples)} eval cases ──\n")

    for i, sample in enumerate(eval_samples):
        prompt_ids = tokenizer.encode(sample["prompt"], return_tensors="pt")

        # Left-truncate prompt to fit in context
        if prompt_ids.shape[1] > MAX_SEQ_LENGTH - 256:
            prompt_ids = prompt_ids[:, -(MAX_SEQ_LENGTH - 256):]

        prompt_ids = prompt_ids.to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=prompt_ids,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
            )

        # Decode only the new tokens
        generated_ids = output_ids[0, prompt_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        scores = score_predictions(generated_text, sample["completion"])

        # Print per-case results
        filename = sample.get("filename", f"case_{i}")
        print(f"  Case: {filename}")
        print(f"    Predicted: {scores['pred_votes']}")
        print(f"    Truth:     {scores['true_votes']}")
        print(f"    Case result correct: {scores['case_correct']}")

        for justice, correct in scores["justice_results"].items():
            if justice not in all_justice_results:
                all_justice_results[justice] = []
            all_justice_results[justice].append(correct)

        if scores["case_correct"] is not None:
            case_results.append(scores["case_correct"])

        print()

    # Summary
    print("── Evaluation Summary ──\n")

    print("  Justice vote accuracy:")
    total_correct = 0
    total_counted = 0
    for justice in sorted(all_justice_results):
        results = all_justice_results[justice]
        scored = [r for r in results if r is not None]
        if scored:
            acc = sum(scored) / len(scored)
            total_correct += sum(scored)
            total_counted += len(scored)
            missing = sum(1 for r in results if r is None)
            extra = f" ({missing} missing)" if missing else ""
            print(f"    {justice:30s}: {acc:5.1%} ({sum(scored)}/{len(scored)}){extra}")
        else:
            print(f"    {justice:30s}: N/A (not predicted)")

    if total_counted:
        print(f"    {'OVERALL':30s}: {total_correct/total_counted:5.1%} "
              f"({total_correct}/{total_counted})")

    print()
    if case_results:
        case_acc = sum(case_results) / len(case_results)
        print(f"  Case result accuracy: {case_acc:5.1%} "
              f"({sum(case_results)}/{len(case_results)})")
    else:
        print("  Case result accuracy: N/A")

    model.config.use_cache = False
    model.train()


# ── Custom Trainer with vote accuracy ─────────────────────────────────────


class VoteAccuracyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        vote_mask = inputs.pop("vote_mask")
        outputs = model(**inputs)
        loss = outputs.loss

        if self.state.global_step % 10 == 0:
            with torch.no_grad():
                preds = outputs.logits.argmax(dim=-1)
                # logits are shifted: logits[i] predicts labels[i+1]
                preds = preds[:, :-1]
                labels = inputs["labels"][:, 1:]
                mask = vote_mask[:, 1:].bool()
                if mask.any():
                    acc = (preds[mask] == labels[mask]).float().mean()
                    self.log({"vote_accuracy": acc.item()})

        return (loss, outputs) if return_outputs else loss


# ── Training ──────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 for SCOTUS vote prediction")
    parser.add_argument(
        "--model", choices=list(MODELS.keys()), default=DEFAULT_MODEL,
        help=f"Model size (default: {DEFAULT_MODEL}). Options: {', '.join(MODELS.keys())}",
    )
    return parser.parse_args()


def train():
    """Main training entry point."""
    args = parse_args()
    os.environ.setdefault("WANDB_PROJECT", "supreme-court")
    model_name = MODELS[args.model]
    model, tokenizer = setup_model_and_tokenizer(model_name)

    # Load and split by year
    all_samples = load_transcripts(DATA_DIR)
    train_samples, eval_samples = split_by_year(all_samples, EVAL_YEAR)

    # Tokenize train set (with 90/10 split for training loss eval)
    train_tokenized = tokenize_and_filter(
        train_samples, tokenizer, MAX_SEQ_LENGTH, "train"
    )
    train_dataset = Dataset.from_list(train_tokenized)
    split = train_dataset.train_test_split(test_size=0.1, seed=42)
    print(f"Train split: {len(split['train'])} train, "
          f"{len(split['test'])} loss-eval samples")

    _base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
    )

    def data_collator(features):
        vote_masks = [f.pop("vote_mask") for f in features]
        batch = _base_collator(features)
        max_len = batch["input_ids"].shape[1]
        batch["vote_mask"] = torch.tensor(
            [vm + [0] * (max_len - len(vm)) for vm in vote_masks]
        )
        return batch

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
        logging_steps=1,
        save_total_limit=2,
        report_to="wandb",
        run_name=f"scotus-{args.model}",
        dataloader_pin_memory=False,
    )

    trainer = VoteAccuracyTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=data_collator,
    )

    trainer.train()

    # Save final adapter
    final_path = os.path.join(OUTPUT_DIR, "final_adapter")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Saved adapter to {final_path}")

    # Run generative evaluation on held-out year
    if eval_samples:
        evaluate_generation(model, tokenizer, eval_samples)
    else:
        print(f"\nNo eval samples found for year {EVAL_YEAR}")


if __name__ == "__main__":
    train()
