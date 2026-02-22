"""
SCOTUS vote prediction: frozen Qwen2.5 feature extraction + per-justice classifier.

Data: case_transcripts_cleaned (transcript text + JUSTICE VOTES block).
The LLM is frozen; only a small head is trained to predict petitioner vs respondent
for each of 9 justice slots.
"""

import argparse
import glob
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = "case_transcripts_cleaned"
VOTES_DELIMITER = "\n---\nJUSTICE VOTES:\n"

# Frozen encoder (Qwen2.5; no fine-tuning)
DEFAULT_LLM = "Qwen/Qwen2.5-0.5B"  # small for fast iteration; use 1.5B/3B/7B for better features
MAX_SEQ_LENGTH = 8192

# 9 justice slots (current court + common aliases for transcript name variants)
JUSTICE_ORDER = [
    "Roberts", "Thomas", "Alito", "Sotomayor", "Kagan",
    "Kavanaugh", "Gorsuch", "Barrett", "Jackson",
]
# Map transcript vote names (lowercase) to canonical slot name (lowercase).
# Only maps to the 9 current slots; older justices (e.g. Ginsburg, Breyer) are skipped.
JUSTICE_ALIASES = {
    "jr.": "roberts",
    "john g. roberts, jr.": "roberts",
    "roberts": "roberts",
    "thomas": "thomas",
    "clarence thomas": "thomas",
    "alito": "alito",
    "samuel a. alito, jr.": "alito",
    "sotomayor": "sotomayor",
    "sonia sotomayor": "sotomayor",
    "kagan": "kagan",
    "elena kagan": "kagan",
    "kavanaugh": "kavanaugh",
    "brett kavanaugh": "kavanaugh",
    "gorsuch": "gorsuch",
    "neil gorsuch": "gorsuch",
    "barrett": "barrett",
    "amy coney barrett": "barrett",
    "jackson": "jackson",
    "ketanji brown jackson": "jackson",
}
# Canonical lowercase for index lookup
JUSTICE_CANONICAL = [j.lower() for j in JUSTICE_ORDER]
NUM_JUSTICES = len(JUSTICE_ORDER)

OUTPUT_DIR = "output/scotus-frozen-head"
EVAL_YEAR = "2019"


# ── Data loading & vote parsing ────────────────────────────────────────────────

def load_transcripts(data_dir: str) -> list[dict]:
    """Load transcript files from case_transcripts_cleaned.

    Prompt = everything up to and including the JUSTICE VOTES: header (fed to LLM).
    Completion = vote lines + outcome (used only for labels).
    """
    samples = []
    paths = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    for path in paths:
        with open(path) as f:
            text = f.read()

        if VOTES_DELIMITER not in text:
            continue

        idx = text.index(VOTES_DELIMITER) + len(VOTES_DELIMITER)
        prompt = text[:idx]
        completion = text[idx:].strip()

        if not completion:
            continue

        samples.append({
            "prompt": prompt,
            "completion": completion,
            "filename": os.path.basename(path),
        })

    print(f"Loaded {len(samples)} samples from {data_dir}")
    return samples


def parse_votes(completion: str) -> dict[str, str]:
    """Parse completion block into {justice_name_lower: 'petitioner'|'respondent'}."""
    out = {}
    for line in completion.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("OUTCOME:"):
            continue
        if ": " not in line:
            continue
        name, side = line.rsplit(": ", 1)
        side = side.strip().lower()
        if side in ("petitioner", "respondent"):
            out[name.strip().lower()] = side
    return out


def votes_to_label_vector(votes: dict[str, str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert parsed votes to (labels, mask) for 9 justice slots.

    labels: (NUM_JUSTICES,) 0 = respondent, 1 = petitioner, -100 = ignore
    mask: (NUM_JUSTICES,) 1 where we have a label, 0 else
    """
    labels = torch.full((NUM_JUSTICES,), -100, dtype=torch.long)
    for name_lower, side in votes.items():
        canonical = JUSTICE_ALIASES.get(name_lower)
        if canonical is None:
            continue
        try:
            idx = JUSTICE_CANONICAL.index(canonical)
        except ValueError:
            continue
        labels[idx] = 1 if side == "petitioner" else 0
    mask = (labels >= 0).long()
    return labels, mask


def split_by_year(samples: list[dict], eval_year: str) -> tuple[list[dict], list[dict]]:
    """Train = not eval_year, Eval = filename starts with eval_year."""
    train_samples = [s for s in samples if not s["filename"].startswith(eval_year)]
    eval_samples = [s for s in samples if s["filename"].startswith(eval_year)]
    print(f"Split: {len(train_samples)} train, {len(eval_samples)} eval (year={eval_year})")
    return train_samples, eval_samples


# ── Dataset ────────────────────────────────────────────────────────────────────

class TranscriptVoteDataset(Dataset):
    """Tokenizes prompts only; labels from completion (per-justice vote)."""

    def __init__(
        self,
        samples: list[dict],
        tokenizer,
        max_length: int = MAX_SEQ_LENGTH,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        enc = self.tokenizer(
            s["prompt"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        votes = parse_votes(s["completion"])
        labels, mask = votes_to_label_vector(votes)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
            "vote_mask": mask,
        }


# ── Model: frozen LLM + vote head ───────────────────────────────────────────────
#
# ARCHITECTURE (detailed)
# ----------------------
# 1. Input: tokenized transcript prompt (B, L) — everything up to "JUSTICE VOTES:"
#    (no vote answers). L ≤ MAX_SEQ_LENGTH; padding on the right.
#
# 2. Frozen Qwen2.5 encoder (causal LM):
#    - Forward pass with output_hidden_states=True.
#    - We take the last layer hidden states: (B, L, H) where H = config.hidden_size
#      (e.g. 896 for 0.5B, 1536 for 1.5B, 3584 for 7B).
#
# 3. Pooling (no learned params):
#    - Mean over the sequence, ignoring padding: for each position t we weight by
#      attention_mask[t]. So we get one vector per case: (B, H).
#
# 4. Post-LLM head (the only trainable part):
#    - Single linear layer: Linear(H, 9*2).
#    - Input: pooled case vector (B, H). Output: (B, 18) interpreted as (B, 9, 2).
#    - Each of the 9 “slots” corresponds to one justice (Roberts, Thomas, …). For each
#      slot we have 2 logits: [respondent, petitioner]. No sharing across justices:
#      the same case vector is mapped to 9 independent binary predictions via one
#      big matrix (no per-justice MLP, no attention).
#
# 5. Loss: cross-entropy for the 2-way choice per justice, only on positions where
#    that justice appears in the label (vote_mask). Justices not in the case (or not
#    in our 9-slot list) are masked out.
#
# So the “post-LLM model” is literally one linear layer: one (H × 18) matrix plus
# bias. No hidden layers, no ReLU, no layer norm.


class FrozenLLMVotePredictor(nn.Module):
    """Frozen causal LM encoder + mean-pooled features -> 9 (petitioner vs respondent) logits."""

    def __init__(self, llm_name: str, num_justices: int = NUM_JUSTICES):
        super().__init__()
        self.encoder = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        for p in self.encoder.parameters():
            p.requires_grad = False
        hidden_size = self.encoder.config.hidden_size
        self.num_justices = num_justices
        # Single linear: (B, H) -> (B, num_justices * 2); no hidden layers
        self.head = nn.Linear(hidden_size, num_justices * 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        with torch.no_grad():
            out = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        hidden = out.hidden_states[-1]  # (B, L, H)
        # Mean pool over non-padding positions
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # (B, H)
        # Head lives on one device; pooled may be on another if encoder uses device_map="auto"
        pooled = pooled.to(next(self.head.parameters()).device).float()
        logits = self.head(pooled)  # (B, num_justices*2)
        return logits.view(logits.size(0), self.num_justices, 2)


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vote_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy per justice, only where vote_mask==1."""
    # logits: (B, 9, 2), labels: (B, 9), vote_mask: (B, 9)
    B, J, _ = logits.shape
    logits_flat = logits.view(-1, 2)
    labels_flat = labels.view(-1)
    mask_flat = vote_mask.view(-1)
    if mask_flat.sum() == 0:
        return logits.sum() * 0.0
    ce = nn.functional.cross_entropy(logits_flat, labels_flat, reduction="none")
    return (ce * mask_flat).sum() / mask_flat.sum().clamp(min=1)


# ── Training & evaluation ───────────────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in tqdm(dataloader, desc="Train"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        vote_mask = batch["vote_mask"].to(device)
        logits = model(input_ids, attention_mask)
        loss = compute_loss(logits, labels, vote_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / n if n else 0.0


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            vote_mask = batch["vote_mask"].to(device)
            logits = model(input_ids, attention_mask)
            loss = compute_loss(logits, labels, vote_mask)
            total_loss += loss.item()
            pred = logits.argmax(dim=-1)  # (B, 9)
            correct += ((pred == labels).long() * vote_mask).sum().item()
            total += vote_mask.sum().item()
    n = len(dataloader)
    return (total_loss / n if n else 0.0), (correct / total if total else 0.0)


def main():
    parser = argparse.ArgumentParser(description="Train vote head on frozen Qwen2.5 features")
    parser.add_argument("--data-dir", default=DATA_DIR, help="case_transcripts_cleaned dir")
    parser.add_argument("--model", default=DEFAULT_LLM, help="Qwen2.5 model name")
    parser.add_argument("--max-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eval-year", default=EVAL_YEAR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    samples = load_transcripts(args.data_dir)
    train_samples, eval_samples = split_by_year(samples, args.eval_year)
    if not train_samples:
        raise RuntimeError("No training samples")
    # Filter to samples that have at least one justice in our 9-slot set
    def has_any_label(s):
        votes = parse_votes(s["completion"])
        _, mask = votes_to_label_vector(votes)
        return mask.sum().item() > 0
    train_samples = [s for s in train_samples if has_any_label(s)]
    eval_samples = [s for s in eval_samples if has_any_label(s)]
    print(f"After filtering: {len(train_samples)} train, {len(eval_samples)} eval")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = TranscriptVoteDataset(train_samples, tokenizer, args.max_length)
    eval_ds = TranscriptVoteDataset(eval_samples, tokenizer, args.max_length)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Model: frozen LLM + trainable head
    model = FrozenLLMVotePredictor(args.model, num_justices=NUM_JUSTICES)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        eval_loss, eval_acc = evaluate(model, eval_loader, device)
        print(f"Epoch {epoch+1}  train_loss={train_loss:.4f}  eval_loss={eval_loss:.4f}  eval_acc={eval_acc:.4f}")

    save_path = os.path.join(args.output_dir, "vote_head.pt")
    torch.save(model.head.state_dict(), save_path)
    print(f"Saved head to {save_path}")


if __name__ == "__main__":
    main()
