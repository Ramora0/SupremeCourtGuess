"""QLoRA fine-tuning of Qwen2.5 for SCOTUS vote prediction."""

import argparse
import glob
import os
import re

import torch
import wandb
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
LEARNING_RATE = 5e-5
NUM_EPOCHS = 1
WARMUP_RATIO = 0.05

DATA_DIR = "case_transcripts_cleaned"
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
    # In context these appear after ": " so the leading space is part of the token
    pet_first = tokenizer.encode(" Petitioner", add_special_tokens=False)[0]
    res_first = tokenizer.encode(" Respondent", add_special_tokens=False)[0]
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
        # Find all vote token positions; last one is the OUTCOME line
        vote_positions = [i for i, lb in enumerate(labels) if lb in vote_tokens]
        vote_mask = [0] * len(labels)
        outcome_mask = [0] * len(labels)

        # Mask all non-vote completion tokens so loss fires only on votes
        vote_position_set = set(vote_positions)
        for i in range(len(prompt_ids), len(labels)):
            if i not in vote_position_set:
                labels[i] = -100
        if vote_positions:
            for pos in vote_positions[:-1]:
                vote_mask[pos] = 1
            outcome_mask[vote_positions[-1]] = 1

        tokenized.append({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "vote_mask": vote_mask,
            "outcome_mask": outcome_mask,
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


def setup_model_and_tokenizer(model_name: str, quantize: bool = True):
    """Load model with SDPA and apply LoRA. Optionally use 4-bit quantization."""
    print(f"Loading model: {model_name} ({'4-bit' if quantize else 'bf16'})")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    if quantize:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **load_kwargs,
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


def evaluate_logits(model, tokenizer, eval_samples: list[dict]):
    """Evaluate with constrained autoregressive generation.

    For each case, feed the prompt via KV cache, then iterate through justice
    lines: force-feed the template tokens (justice name + colon), let the model
    predict the vote token (Petitioner/Respondent), feed the predicted vote's
    full token sequence back, then continue to the next justice.  The model's
    own predictions flow through the context — no teacher forcing.
    """
    model.eval()
    model.config.use_cache = True

    pet_first = tokenizer.encode(" Petitioner", add_special_tokens=False)[0]
    res_first = tokenizer.encode(" Respondent", add_special_tokens=False)[0]
    pet_all = tokenizer.encode(" Petitioner", add_special_tokens=False)
    res_all = tokenizer.encode(" Respondent", add_special_tokens=False)
    vote_token_set = {pet_first, res_first}

    all_justice_results = {}  # justice -> list of bool
    case_results = []

    print(f"\n── Logit-based evaluation for {len(eval_samples)} eval cases ──\n")

    for i, sample in enumerate(eval_samples):
        # Tokenize prompt and completion separately
        prompt_ids = tokenizer.encode(sample["prompt"], add_special_tokens=False)
        completion_ids = tokenizer.encode(sample["completion"], add_special_tokens=False)

        if len(prompt_ids) > MAX_SEQ_LENGTH - 256:
            prompt_ids = prompt_ids[-(MAX_SEQ_LENGTH - 256):]

        # Find vote token positions within completion_ids
        vote_positions = [j for j, tid in enumerate(completion_ids)
                          if tid in vote_token_set]

        # Parse completion text to get ordered justice names
        justice_names_ordered = []
        for line in sample["completion"].strip().splitlines():
            line = line.strip()
            if not line or line.startswith("OUTCOME:"):
                continue
            if ": " not in line:
                continue
            name, side = line.rsplit(": ", 1)
            if side.strip().lower() in ("petitioner", "respondent"):
                justice_names_ordered.append(name.strip().lower())

        # Justice votes are all positions except the last (OUTCOME line)
        jvp = (vote_positions[:-1]
               if len(vote_positions) > len(justice_names_ordered)
               else vote_positions[:len(justice_names_ordered)])

        assert len(jvp) == len(justice_names_ordered), (
            f"Case {sample.get('filename', i)}: {len(jvp)} vote positions "
            f"!= {len(justice_names_ordered)} justices"
        )

        # Feed prompt through model, build KV cache
        input_tensor = torch.tensor([prompt_ids], device=model.device)
        with torch.no_grad():
            out = model(input_ids=input_tensor, use_cache=True)
            past_kv = out.past_key_values

        pred_votes = {}
        prev_end = 0  # cursor into completion_ids for template segments

        for j, (vpos, justice) in enumerate(zip(jvp, justice_names_ordered)):
            # Feed template tokens before this vote (justice name, colon, etc.)
            template_ids = completion_ids[prev_end:vpos]
            if template_ids:
                t = torch.tensor([template_ids], device=model.device)
                with torch.no_grad():
                    out = model(input_ids=t, past_key_values=past_kv,
                                use_cache=True)
                    past_kv = out.past_key_values

            # Read logit at last position — predicts the vote first subtoken
            last_logit = out.logits[0, -1, :]
            pet_logit = last_logit[pet_first].item()
            res_logit = last_logit[res_first].item()
            pred_side = "petitioner" if pet_logit > res_logit else "respondent"
            pred_votes[justice] = pred_side

            # Feed the PREDICTED vote's full token sequence into KV cache
            vote_full = pet_all if pred_side == "petitioner" else res_all
            v = torch.tensor([vote_full], device=model.device)
            with torch.no_grad():
                out = model(input_ids=v, past_key_values=past_kv,
                            use_cache=True)
                past_kv = out.past_key_values

            # Advance cursor past the ground-truth vote tokens in completion_ids
            gt_vote = pet_all if completion_ids[vpos] == pet_first else res_all
            prev_end = vpos + len(gt_vote)

        # Score against ground truth
        true_votes = parse_votes(sample["completion"])
        for justice in justice_names_ordered:
            if justice in true_votes:
                correct = pred_votes[justice] == true_votes[justice]
                if justice not in all_justice_results:
                    all_justice_results[justice] = []
                all_justice_results[justice].append(correct)

        pred_result = case_result(pred_votes)
        true_result = case_result(true_votes)
        case_correct = pred_result == true_result if true_result else None

        filename = sample.get("filename", f"case_{i}")
        print(f"  Case: {filename}")
        print(f"    Predicted: {pred_votes}")
        print(f"    Truth:     {true_votes}")
        print(f"    Case result correct: {case_correct}")

        if case_correct is not None:
            case_results.append(case_correct)

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
            print(f"    {justice:30s}: {acc:5.1%} ({sum(scored)}/{len(scored)})")
        else:
            print(f"    {justice:30s}: N/A")

    if total_counted:
        print(f"    {'OVERALL':30s}: {total_correct/total_counted:5.1%} "
              f"({total_correct}/{total_counted})")

    print()
    eval_metrics = {}
    if total_counted:
        eval_metrics["eval/justice_vote_acc"] = total_correct / total_counted
        for justice in sorted(all_justice_results):
            scored = [r for r in all_justice_results[justice] if r is not None]
            if scored:
                eval_metrics[f"eval/justice/{justice}"] = sum(scored) / len(scored)
    if case_results:
        case_acc = sum(case_results) / len(case_results)
        eval_metrics["eval/case_acc"] = case_acc
        print(f"  Case result accuracy: {case_acc:5.1%} "
              f"({sum(case_results)}/{len(case_results)})")
    else:
        print("  Case result accuracy: N/A")

    if eval_metrics and wandb.run is not None:
        wandb.log(eval_metrics)

    model.config.use_cache = False
    model.train()


# ── Custom Trainer with vote accuracy ─────────────────────────────────────


class VoteAccuracyTrainer(Trainer):
    def __init__(self, *args, vote_token_ids=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._vote_token_ids = vote_token_ids  # [pet_first, res_first]
        self._vote_prob_sum = 0.0
        self._vote_total = 0
        self._outcome_prob_sum = 0.0
        self._outcome_total = 0
        self._format_prob_sum = 0.0
        self._format_total = 0
        self._majority_prob_sum = 0.0
        self._majority_cases = 0
        self._first_vote_prob_sum = 0.0
        self._first_vote_total = 0
        self._micro_step = 0

    @staticmethod
    def _majority_correct_prob(probs_correct):
        """P(>=ceil(n/2) correct) via DP over per-justice correct probs."""
        n = len(probs_correct)
        # dp[k] = P(exactly k correct so far)
        dp = [0.0] * (n + 1)
        dp[0] = 1.0
        for p in probs_correct:
            new_dp = [0.0] * (n + 1)
            for k in range(n + 1):
                if dp[k] == 0.0:
                    continue
                if k + 1 <= n:
                    new_dp[k + 1] += dp[k] * p
                new_dp[k] += dp[k] * (1.0 - p)
            dp = new_dp
        majority = (n // 2) + 1
        return sum(dp[majority:])

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        vote_mask = inputs.pop("vote_mask")
        outcome_mask = inputs.pop("outcome_mask")
        pet_id, res_id = self._vote_token_ids
        outputs = model(**inputs)

        # Binary cross-entropy over only the two vote token logits
        # Shift: logits[i] predicts labels[i+1]
        shifted_logits = outputs.logits[:, :-1, :]
        shifted_labels = inputs["labels"][:, 1:]
        shifted_vote_mask = vote_mask[:, 1:].bool()

        if shifted_vote_mask.any():
            vote_logits = shifted_logits[shifted_vote_mask][:, [pet_id, res_id]]  # (N, 2)
            vote_labels = shifted_labels[shifted_vote_mask]
            # Map label token IDs to binary targets: 0 = petitioner, 1 = respondent
            binary_targets = (vote_labels == res_id).long()
            loss = torch.nn.functional.cross_entropy(vote_logits, binary_targets)
        else:
            loss = outputs.loss  # fallback if no vote tokens found

        self._micro_step += 1

        with torch.no_grad():
            probs = shifted_logits.softmax(dim=-1)

            v_mask = shifted_vote_mask
            if v_mask.any():
                label_ids = shifted_labels[v_mask]
                correct_probs = probs[v_mask].gather(1, label_ids.unsqueeze(1)).squeeze(1)
                self._vote_prob_sum += correct_probs.sum().item()
                self._vote_total += v_mask.sum().item()

                # First justice vote prob (no teacher forcing advantage)
                first_vote_probs = probs[v_mask][0]
                self._first_vote_prob_sum += correct_probs[0].item()
                self._format_prob_sum += first_vote_probs[self._vote_token_ids].sum().item()
                self._format_total += 1
                self._first_vote_total += 1

                # P(majority correct) per case via Poisson binomial DP
                # batch_size=1, so correct_probs is all justices for one case
                self._majority_prob_sum += self._majority_correct_prob(
                    correct_probs.tolist()
                )
                self._majority_cases += 1

            o_mask = outcome_mask[:, 1:].bool()
            if o_mask.any():
                label_ids = labels[o_mask]
                correct_probs = probs[o_mask].gather(1, label_ids.unsqueeze(1)).squeeze(1)
                self._outcome_prob_sum += correct_probs.sum().item()
                self._outcome_total += o_mask.sum().item()

        # Log only after full grad accumulation window
        if self._micro_step % self.args.gradient_accumulation_steps == 0:
            metrics = {}
            if self._vote_total > 0:
                metrics["vote_prob"] = self._vote_prob_sum / self._vote_total
            if self._outcome_total > 0:
                metrics["outcome_prob"] = self._outcome_prob_sum / self._outcome_total
            if self._format_total > 0:
                metrics["format_prob"] = self._format_prob_sum / self._format_total
            if self._majority_cases > 0:
                metrics["majority_correct_prob"] = self._majority_prob_sum / self._majority_cases
            if self._first_vote_total > 0:
                metrics["first_vote_prob"] = self._first_vote_prob_sum / self._first_vote_total
            if metrics:
                self.log(metrics)
            self._vote_prob_sum = 0.0
            self._vote_total = 0
            self._outcome_prob_sum = 0.0
            self._outcome_total = 0
            self._format_prob_sum = 0.0
            self._format_total = 0
            self._majority_prob_sum = 0.0
            self._majority_cases = 0
            self._first_vote_prob_sum = 0.0
            self._first_vote_total = 0

        return (loss, outputs) if return_outputs else loss


# ── Training ──────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 for SCOTUS vote prediction")
    parser.add_argument(
        "--model", choices=list(MODELS.keys()), default=DEFAULT_MODEL,
        help=f"Model size (default: {DEFAULT_MODEL}). Options: {', '.join(MODELS.keys())}",
    )
    parser.add_argument(
        "--no-quant", action="store_true",
        help="Disable 4-bit quantization (use bf16). Required for multi-GPU DDP training.",
    )
    return parser.parse_args()


def train():
    """Main training entry point."""
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0
    os.environ.setdefault("WANDB_PROJECT", "supreme-court")
    model_name = MODELS[args.model]
    model, tokenizer = setup_model_and_tokenizer(model_name, quantize=not args.no_quant)

    # Load and split by year
    all_samples = load_transcripts(DATA_DIR)
    train_samples, eval_samples = split_by_year(all_samples, EVAL_YEAR)

    # Tokenize train set (with 90/10 split for training loss eval)
    train_tokenized = tokenize_and_filter(
        train_samples, tokenizer, MAX_SEQ_LENGTH, "train"
    )
    train_dataset = Dataset.from_list(train_tokenized)
    split = train_dataset.train_test_split(test_size=0.1, seed=42)
    if is_main:
        print(f"Train split: {len(split['train'])} train, "
              f"{len(split['test'])} loss-eval samples")

    _base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
    )

    def data_collator(features):
        vote_masks = [f.pop("vote_mask") for f in features]
        outcome_masks = [f.pop("outcome_mask") for f in features]
        batch = _base_collator(features)
        max_len = batch["input_ids"].shape[1]
        batch["vote_mask"] = torch.tensor(
            [vm + [0] * (max_len - len(vm)) for vm in vote_masks]
        )
        batch["outcome_mask"] = torch.tensor(
            [om + [0] * (max_len - len(om)) for om in outcome_masks]
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
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=1,
        save_total_limit=2,
        report_to="wandb" if is_main else "none",
        run_name=f"scotus-{args.model}",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    pet_first = tokenizer.encode(" Petitioner", add_special_tokens=False)[0]
    res_first = tokenizer.encode(" Respondent", add_special_tokens=False)[0]
    trainer = VoteAccuracyTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=data_collator,
        vote_token_ids=[pet_first, res_first],
    )

    trainer.train()

    # Save final adapter (only on main process)
    if is_main:
        final_path = os.path.join(OUTPUT_DIR, "final_adapter")
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"Saved adapter to {final_path}")

        # Run generative evaluation on held-out year
        if eval_samples:
            evaluate_logits(model, tokenizer, eval_samples)
        else:
            print(f"\nNo eval samples found for year {EVAL_YEAR}")


if __name__ == "__main__":
    train()
