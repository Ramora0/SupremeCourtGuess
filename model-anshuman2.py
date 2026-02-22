"""
SCOTUS vote prediction (v2): frozen Qwen2.5 + cross-attention head.

Same data and labels as model-anshuman.py. The post-LLM predictor is a
cross-attention layer: 9 learnable justice queries attend over the encoder
hidden states (no pooling). Output per justice is then projected to 2-way
logits (respondent / petitioner) and trained with cross-entropy.
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

DEFAULT_LLM = "Qwen/Qwen2.5-0.5B"
MAX_SEQ_LENGTH = 8192

JUSTICE_ORDER = [
    "Roberts", "Thomas", "Alito", "Sotomayor", "Kagan",
    "Kavanaugh", "Gorsuch", "Barrett", "Jackson",
]
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
JUSTICE_CANONICAL = [j.lower() for j in JUSTICE_ORDER]
NUM_JUSTICES = len(JUSTICE_ORDER)

OUTPUT_DIR = "output/scotus-frozen-crossattn"
EVAL_YEAR = "2019"


# ── Data loading & vote parsing ────────────────────────────────────────────────

def load_transcripts(data_dir: str) -> list[dict]:
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
        samples.append({"prompt": prompt, "completion": completion, "filename": os.path.basename(path)})
    print(f"Loaded {len(samples)} samples from {data_dir}")
    return samples


def parse_votes(completion: str) -> dict[str, str]:
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
    train_samples = [s for s in samples if not s["filename"].startswith(eval_year)]
    eval_samples = [s for s in samples if s["filename"].startswith(eval_year)]
    print(f"Split: {len(train_samples)} train, {len(eval_samples)} eval (year={eval_year})")
    return train_samples, eval_samples


# ── Dataset ────────────────────────────────────────────────────────────────────

class TranscriptVoteDataset(Dataset):
    def __init__(self, samples: list[dict], tokenizer, max_length: int = MAX_SEQ_LENGTH):
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


# ── Model: frozen LLM + cross-attention vote head ──────────────────────────────
#
# ARCHITECTURE (v2 — cross-attention)
# ----------------------------------
# 1. Input: tokenized prompt (B, L). Frozen Qwen2.5 returns last-layer hidden (B, L, H).
#
# 2. No pooling. We keep the full feature matrix (B, L, H) as keys/values.
#
# 3. Cross-attention head:
#    - 9 learnable query vectors: Q_learned (9, H), one per justice.
#    - Keys K = hidden (B, L, H), Values V = hidden (B, L, H).
#    - For each batch: Q = Q_learned expanded to (B, 9, H).
#    - Scores = Q @ K^T → (B, 9, L). Mask padding positions with -inf.
#    - Attention weights = softmax(scores, dim=-1) over the sequence (L).
#    - Output = attn_weights @ V → (B, 9, H). So each justice gets a weighted
#      combination of encoder positions.
#
# 4. Vote logits: Linear(H, 2) applied to each of the 9 outputs → (B, 9, 2).
#    Loss: cross-entropy (respondent vs petitioner) per justice, masked as in v1.


class CrossAttentionVoteHead(nn.Module):
    """9 justice queries cross-attend over (B, L, H); then linear to (B, 9, 2)."""

    def __init__(self, hidden_size: int, num_justices: int = NUM_JUSTICES):
        super().__init__()
        self.num_justices = num_justices
        self.scale = hidden_size ** -0.5
        # Learnable queries: one vector per justice (9, H)
        self.justice_queries = nn.Parameter(torch.randn(num_justices, hidden_size) * 0.02)
        # After cross-attention we have (B, 9, H); project each to 2 logits (shared linear)
        self.vote_proj = nn.Linear(hidden_size, 2)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # hidden: (B, L, H), attention_mask: (B, L) 1=real, 0=pad
        B, L, H = hidden.shape
        device = hidden.device
        dtype = hidden.dtype
        # Move to float for attention if needed
        hidden = hidden.float()
        Q = self.justice_queries.unsqueeze(0).expand(B, -1, -1)  # (B, 9, H)
        K = hidden  # (B, L, H)
        V = hidden  # (B, L, H)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, 9, L)
        # Mask padding: where attention_mask == 0, set score to -inf
        pad_mask = (attention_mask == 0).unsqueeze(1)  # (B, 1, L)
        scores = scores.masked_fill(pad_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)  # (B, 9, L)
        out = torch.matmul(attn, V)  # (B, 9, H)
        logits = self.vote_proj(out)  # (B, 9, 2)
        return logits


class FrozenLLMVotePredictorCrossAttn(nn.Module):
    """Frozen Qwen2.5 encoder + cross-attention head (no mean pooling)."""

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
        self.head = CrossAttentionVoteHead(hidden_size, num_justices=num_justices)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        with torch.no_grad():
            out = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        hidden = out.hidden_states[-1]  # (B, L, H)
        # Head may live on a different device (e.g. cuda:0 when encoder is device_map="auto")
        head_device = next(self.head.parameters()).device
        hidden = hidden.to(head_device)
        attention_mask = attention_mask.to(head_device)
        logits = self.head(hidden, attention_mask)  # (B, 9, 2)
        return logits


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vote_mask: torch.Tensor,
) -> torch.Tensor:
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
            pred = logits.argmax(dim=-1)
            correct += ((pred == labels).long() * vote_mask).sum().item()
            total += vote_mask.sum().item()
    n = len(dataloader)
    return (total_loss / n if n else 0.0), (correct / total if total else 0.0)


def main():
    parser = argparse.ArgumentParser(description="Train cross-attention vote head on frozen Qwen2.5")
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

    samples = load_transcripts(args.data_dir)
    train_samples, eval_samples = split_by_year(samples, args.eval_year)
    if not train_samples:
        raise RuntimeError("No training samples")

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
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = FrozenLLMVotePredictorCrossAttn(args.model, num_justices=NUM_JUSTICES)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        eval_loss, eval_acc = evaluate(model, eval_loader, device)
        print(f"Epoch {epoch+1}  train_loss={train_loss:.4f}  eval_loss={eval_loss:.4f}  eval_acc={eval_acc:.4f}")

    save_path = os.path.join(args.output_dir, "vote_head_crossattn.pt")
    torch.save(model.head.state_dict(), save_path)
    print(f"Saved head to {save_path}")


if __name__ == "__main__":
    main()
