"""Cross-attention SCOTUS vote predictor with frozen Qwen2.5 backbone."""

import argparse
import glob
import math
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Config ────────────────────────────────────────────────────────────────────

MODELS = {
    "7b": "Qwen/Qwen2.5-7B",
    "3b": "Qwen/Qwen2.5-3B",
    "1.5b": "Qwen/Qwen2.5-1.5B",
    "0.5b": "Qwen/Qwen2.5-0.5B",
}
DEFAULT_MODEL = "7b"

MAX_SEQ_LENGTH = 16384
DATA_DIR = "case_transcripts_clean"
VOTES_DELIMITER = "\n---\nJUSTICE VOTES:\n"
EVAL_YEAR = "2019"

# Training
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
GRAD_ACCUM_STEPS = 8
NUM_EPOCHS = 3
HEAD_DIM = 256
NUM_QUERIES_PER_JUSTICE = 8
SELF_ATTN_LAYERS = 2
SELF_ATTN_FFN_DIM = 1024
MAX_JUSTICES = 128


# ── Data loading ──────────────────────────────────────────────────────────────


def load_transcripts(data_dir: str) -> list[dict]:
    """Load transcript files and split into transcript text and vote labels."""
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

        idx = text.index(VOTES_DELIMITER)
        transcript = text[:idx]
        votes_text = text[idx + len(VOTES_DELIMITER):].strip()

        if not votes_text:
            print(f"Warning: skipping {path} — empty votes")
            continue

        votes = parse_votes(votes_text)
        if not votes:
            print(f"Warning: skipping {path} — no parseable votes")
            continue

        filename = os.path.basename(path)
        samples.append({
            "transcript": transcript,
            "votes": votes,
            "filename": filename,
        })

    print(f"Loaded {len(samples)} samples from {data_dir}")
    return samples


def parse_votes(text: str) -> dict[str, int]:
    """Parse vote lines into {justice_name: 0|1} (0=Petitioner, 1=Respondent)."""
    votes = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("OUTCOME:"):
            continue
        if ": " not in line:
            continue
        name, side = line.rsplit(": ", 1)
        side = side.strip().lower()
        if side == "petitioner":
            votes[name.strip()] = 0
        elif side == "respondent":
            votes[name.strip()] = 1
    return votes


def split_by_year(
    samples: list[dict], eval_year: str
) -> tuple[list[dict], list[dict]]:
    """Split samples into train/eval based on filename year prefix."""
    train_samples, eval_samples = [], []
    for s in samples:
        if s["filename"].startswith(eval_year):
            eval_samples.append(s)
        else:
            train_samples.append(s)
    print(f"Year split: {len(train_samples)} train, "
          f"{len(eval_samples)} eval (year={eval_year})")
    return train_samples, eval_samples


def case_result(votes: dict[str, int]) -> int | None:
    """Determine case winner by majority vote. 0=Petitioner, 1=Respondent."""
    if not votes:
        return None
    pet = sum(1 for v in votes.values() if v == 0)
    resp = sum(1 for v in votes.values() if v == 1)
    if pet > resp:
        return 0
    elif resp > pet:
        return 1
    return None


# ── Justice Registry ──────────────────────────────────────────────────────────


class JusticeRegistry:
    """Maps justice names to integer IDs, discovered from data."""

    def __init__(self):
        self.name_to_id: dict[str, int] = {}
        self.id_to_name: dict[int, str] = {}
        self._next_id = 0

    def get_or_add(self, name: str) -> int:
        if name not in self.name_to_id:
            self.name_to_id[name] = self._next_id
            self.id_to_name[self._next_id] = name
            self._next_id += 1
        return self.name_to_id[name]

    def build_from_samples(self, samples: list[dict]):
        for s in samples:
            for name in s["votes"]:
                self.get_or_add(name)
        print(f"Justice registry: {len(self.name_to_id)} unique justices")

    def __len__(self):
        return len(self.name_to_id)


# ── Model Components ──────────────────────────────────────────────────────────


class LayerWeighter(nn.Module):
    """Learns softmax weights over LLM hidden layers (ELMo-style)."""

    def __init__(self, num_layers: int):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))

    def forward(self, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        """Weighted sum of hidden states.

        Args:
            hidden_states: list of (1, S, D) tensors (detached)
        Returns:
            blended: (1, S, D)
        """
        weights = F.softmax(self.layer_logits, dim=0)
        stacked = torch.stack(hidden_states, dim=0)  # (L, 1, S, D)
        blended = torch.einsum("l,lbsd->bsd", weights, stacked)
        return blended


class JusticeEmbeddings(nn.Module):
    """Learnable query vectors per justice."""

    def __init__(self, max_justices: int, num_queries: int, dim: int):
        super().__init__()
        self.num_queries = num_queries
        self.embedding = nn.Embedding(max_justices * num_queries, dim)

    def forward(self, justice_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            justice_ids: (J,) tensor of justice integer IDs
        Returns:
            queries: (J, num_queries, dim)
        """
        J = justice_ids.shape[0]
        # Each justice gets num_queries consecutive embedding rows
        base = justice_ids.unsqueeze(1) * self.num_queries  # (J, 1)
        offsets = torch.arange(self.num_queries, device=justice_ids.device)  # (Q,)
        indices = base + offsets  # (J, Q)
        return self.embedding(indices)  # (J, Q, dim)


class TranscriptCrossAttention(nn.Module):
    """Cross-attention from justice queries to transcript hidden states."""

    def __init__(self, llm_dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        # Project LLM hidden states to K and V
        self.kv_proj = nn.Linear(llm_dim, inner_dim * 2, bias=False)
        # Q comes from justice embeddings (already head_dim * num_queries)
        # We treat num_queries == num_heads, each query vector is one head's Q
        self.out_proj = nn.Linear(inner_dim, inner_dim, bias=False)
        self.layer_norm = nn.LayerNorm(inner_dim)

    def forward(
        self, justice_queries: torch.Tensor, transcript: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            justice_queries: (J, num_heads, head_dim)
            transcript: (1, S, llm_dim)
        Returns:
            output: (J, num_heads, head_dim)
        """
        J, H, D = justice_queries.shape
        S = transcript.shape[1]

        # Project transcript to K, V
        kv = self.kv_proj(transcript.squeeze(0))  # (S, H*D*2)
        k, v = kv.chunk(2, dim=-1)  # each (S, H*D)
        k = k.view(S, H, D).permute(1, 0, 2)  # (H, S, D)
        v = v.view(S, H, D).permute(1, 0, 2)  # (H, S, D)

        # Prepare Q: (J, H, D) -> (J*H, 1, D) -> reshape for multi-head
        q = justice_queries  # (J, H, D)

        # Use scaled_dot_product_attention per justice
        # Expand K, V for all justices: (H, S, D) -> (J*H, S, D)
        k = k.unsqueeze(0).expand(J, -1, -1, -1).reshape(J * H, S, D)
        v = v.unsqueeze(0).expand(J, -1, -1, -1).reshape(J * H, S, D)
        q = q.reshape(J * H, 1, D)  # (J*H, 1, D)

        attn_out = F.scaled_dot_product_attention(q, k, v)  # (J*H, 1, D)
        attn_out = attn_out.view(J, H, D)  # (J, H, D)

        # Output projection + residual + layer norm
        flat = attn_out.view(J, H * D)  # (J, H*D)
        out = self.out_proj(flat).view(J, H, D)
        out = self.layer_norm((out + justice_queries).view(J, H * D)).view(J, H, D)
        return out


class VoteClassifier(nn.Module):
    """Pool per-justice vectors and classify vote."""

    def __init__(self, num_queries: int, dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.classifier = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 2),
        )

    def forward(self, justice_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            justice_repr: (J, num_queries, dim)
        Returns:
            logits: (J, 2)
        """
        pooled = justice_repr.mean(dim=1)  # (J, dim)
        normed = self.layer_norm(pooled)
        return self.classifier(normed)


class SCOTUSVoteHead(nn.Module):
    """Complete voting head: layer weighting + cross-attention + self-attention + classifier."""

    def __init__(
        self,
        num_llm_layers: int,
        llm_dim: int,
        max_justices: int = MAX_JUSTICES,
        head_dim: int = HEAD_DIM,
        num_queries: int = NUM_QUERIES_PER_JUSTICE,
        self_attn_layers: int = SELF_ATTN_LAYERS,
        self_attn_ffn_dim: int = SELF_ATTN_FFN_DIM,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.head_dim = head_dim

        self.layer_weighter = LayerWeighter(num_llm_layers)
        self.justice_embeddings = JusticeEmbeddings(max_justices, num_queries, head_dim)
        self.cross_attention = TranscriptCrossAttention(llm_dim, head_dim, num_queries)

        inner_dim = num_queries * head_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=inner_dim,
            nhead=num_queries,
            dim_feedforward=self_attn_ffn_dim,
            batch_first=True,
            norm_first=True,
        )
        self.self_attention = nn.TransformerEncoder(
            encoder_layer, num_layers=self_attn_layers
        )
        self.vote_classifier = VoteClassifier(num_queries, head_dim)

    def forward(
        self,
        blended: torch.Tensor,
        justice_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            blended: (1, S, D) pre-blended transcript hidden states
            justice_ids: (J,) integer justice IDs
        Returns:
            logits: (J, 2) vote logits per justice
        """
        # 1. Get justice query vectors
        queries = self.justice_embeddings(justice_ids)  # (J, Q, head_dim)

        # 3. Cross-attend to transcript
        cross_out = self.cross_attention(queries, blended)  # (J, Q, head_dim)

        # 4. Self-attention across justices
        J = cross_out.shape[0]
        flat = cross_out.view(J, -1).unsqueeze(0)  # (1, J, Q*head_dim)
        self_out = self.self_attention(flat)  # (1, J, Q*head_dim)
        self_out = self_out.squeeze(0).view(J, self.num_queries, self.head_dim)

        # 5. Classify
        logits = self.vote_classifier(self_out)  # (J, 2)
        return logits

    def get_layer_weights(self) -> torch.Tensor:
        """Return current softmax layer weights for logging."""
        with torch.no_grad():
            return F.softmax(self.layer_weighter.layer_logits, dim=0)


# ── Backbone setup ────────────────────────────────────────────────────────────


def load_frozen_backbone(model_name: str, quantize: bool = True):
    """Load Qwen2.5 in 4-bit, fully frozen, with hidden state output."""
    print(f"Loading frozen backbone: {model_name} ({'4-bit' if quantize else 'bf16'})")
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

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.config.use_cache = False

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    return model, tokenizer


def get_blended_hidden_state(
    backbone,
    tokenizer,
    text: str,
    max_length: int,
    device: torch.device,
    layer_weighter: LayerWeighter,
) -> torch.Tensor:
    """Run frozen backbone, blend hidden states immediately, free the rest.

    This keeps only one (1, S, D) tensor alive instead of all 29 layers,
    cutting peak hidden-state memory from ~6.8GB to ~234MB (for 7B, 16K seq).
    """
    input_ids = tokenizer.encode(text, add_special_tokens=False)

    # Left-truncate to max_length
    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]

    input_tensor = torch.tensor([input_ids], device=device)

    with torch.no_grad():
        outputs = backbone(input_ids=input_tensor, output_hidden_states=True)

    # Detach and convert to float32, then blend immediately
    hidden_states = [h.detach().float() for h in outputs.hidden_states]
    del outputs

    blended = layer_weighter(hidden_states)  # (1, S, D)

    # Free the 29 individual hidden state tensors
    del hidden_states
    torch.cuda.empty_cache()

    return blended


# ── Training ──────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train cross-attention SCOTUS vote predictor"
    )
    parser.add_argument(
        "--model", choices=list(MODELS.keys()), default=DEFAULT_MODEL,
        help=f"Backbone model size (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--no-quant", action="store_true",
        help="Disable 4-bit quantization (use bf16)",
    )
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS,
        help=f"Number of training epochs (default: {NUM_EPOCHS})",
    )
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})",
    )
    return parser.parse_args()


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.environ.setdefault("WANDB_PROJECT", "supreme-court")
    wandb.init(
        project="supreme-court",
        name=f"scotus-crossattn-{args.model}",
        config={
            "model": args.model,
            "lr": args.lr,
            "epochs": args.epochs,
            "grad_accum": GRAD_ACCUM_STEPS,
            "head_dim": HEAD_DIM,
            "num_queries": NUM_QUERIES_PER_JUSTICE,
            "self_attn_layers": SELF_ATTN_LAYERS,
            "max_seq_length": MAX_SEQ_LENGTH,
        },
    )

    # Load data
    all_samples = load_transcripts(DATA_DIR)
    train_samples, eval_samples = split_by_year(all_samples, EVAL_YEAR)

    # Build justice registry from ALL data so eval justices are known
    registry = JusticeRegistry()
    registry.build_from_samples(all_samples)

    # Load frozen backbone
    model_name = MODELS[args.model]
    backbone, tokenizer = load_frozen_backbone(model_name, quantize=not args.no_quant)

    # Get model dimensions from config
    num_llm_layers = backbone.config.num_hidden_layers + 1  # +1 for embedding layer
    llm_dim = backbone.config.hidden_size
    print(f"Backbone: {num_llm_layers} hidden state layers, dim={llm_dim}")

    # Build vote head
    head = SCOTUSVoteHead(
        num_llm_layers=num_llm_layers,
        llm_dim=llm_dim,
    ).to(device).float()

    total_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"Trainable head parameters: {total_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY
    )
    total_steps = (len(train_samples) * args.epochs) // GRAD_ACCUM_STEPS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )

    # Training loop
    global_step = 0
    accum_loss = 0.0
    accum_correct = 0
    accum_total = 0
    accum_case_correct = 0
    accum_cases = 0

    for epoch in range(args.epochs):
        head.train()
        pbar = tqdm(train_samples, desc=f"Epoch {epoch+1}/{args.epochs}")

        for i, sample in enumerate(pbar):
            # Get hidden states from frozen backbone
            hidden_states = get_hidden_states(
                backbone, tokenizer, sample["transcript"],
                MAX_SEQ_LENGTH, device,
            )

            # Prepare justice IDs and labels
            justice_names = list(sample["votes"].keys())
            justice_ids = torch.tensor(
                [registry.get_or_add(n) for n in justice_names],
                device=device,
            )
            labels = torch.tensor(
                [sample["votes"][n] for n in justice_names],
                device=device, dtype=torch.long,
            )

            # Forward through head
            logits = head(hidden_states, justice_ids)  # (J, 2)

            # Free hidden states immediately
            del hidden_states

            # Loss averaged over justices
            loss = F.cross_entropy(logits, labels) / GRAD_ACCUM_STEPS

            loss.backward()

            # Track metrics
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                correct = (preds == labels).sum().item()
                accum_correct += correct
                accum_total += len(labels)
                accum_loss += loss.item() * GRAD_ACCUM_STEPS

                # Case accuracy (majority vote)
                pred_majority = 0 if (preds == 0).sum() > (preds == 1).sum() else 1
                true_majority = case_result(sample["votes"])
                if true_majority is not None:
                    accum_case_correct += int(pred_majority == true_majority)
                    accum_cases += 1

            # Gradient accumulation step
            if (i + 1) % GRAD_ACCUM_STEPS == 0 or (i + 1) == len(train_samples):
                nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log to wandb
                layer_weights = head.get_layer_weights()
                metrics = {
                    "train/loss": accum_loss / GRAD_ACCUM_STEPS,
                    "train/vote_acc": accum_correct / max(accum_total, 1),
                    "train/case_acc": accum_case_correct / max(accum_cases, 1),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/step": global_step,
                }
                # Log top-5 layer weights
                top_vals, top_idx = layer_weights.topk(5)
                for rank, (val, idx) in enumerate(zip(top_vals, top_idx)):
                    metrics[f"layers/top{rank+1}_layer"] = idx.item()
                    metrics[f"layers/top{rank+1}_weight"] = val.item()

                wandb.log(metrics, step=global_step)

                pbar.set_postfix(
                    loss=f"{accum_loss / GRAD_ACCUM_STEPS:.4f}",
                    vacc=f"{accum_correct / max(accum_total, 1):.3f}",
                    cacc=f"{accum_case_correct / max(accum_cases, 1):.3f}",
                )

                accum_loss = 0.0
                accum_correct = 0
                accum_total = 0
                accum_case_correct = 0
                accum_cases = 0

        # End of epoch evaluation
        if eval_samples:
            evaluate(head, backbone, tokenizer, eval_samples, registry, device,
                     global_step, epoch)

    # Save head weights
    save_path = f"output/scotus-crossattn-{args.model}.pt"
    os.makedirs("output", exist_ok=True)
    torch.save({
        "head_state_dict": head.state_dict(),
        "registry": registry.name_to_id,
        "config": {
            "num_llm_layers": num_llm_layers,
            "llm_dim": llm_dim,
            "head_dim": HEAD_DIM,
            "num_queries": NUM_QUERIES_PER_JUSTICE,
            "self_attn_layers": SELF_ATTN_LAYERS,
            "self_attn_ffn_dim": SELF_ATTN_FFN_DIM,
            "max_justices": MAX_JUSTICES,
        },
    }, save_path)
    print(f"Saved head to {save_path}")

    wandb.finish()


# ── Evaluation ────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(
    head: SCOTUSVoteHead,
    backbone,
    tokenizer,
    eval_samples: list[dict],
    registry: JusticeRegistry,
    device: torch.device,
    global_step: int,
    epoch: int,
):
    head.eval()
    all_justice_results: dict[str, list[bool]] = {}
    case_results_list: list[bool] = []

    print(f"\n── Evaluation ({len(eval_samples)} cases, epoch {epoch+1}) ──\n")

    for sample in tqdm(eval_samples, desc="Evaluating"):
        hidden_states = get_hidden_states(
            backbone, tokenizer, sample["transcript"],
            MAX_SEQ_LENGTH, device,
        )

        justice_names = list(sample["votes"].keys())
        justice_ids = torch.tensor(
            [registry.get_or_add(n) for n in justice_names],
            device=device,
        )
        labels = torch.tensor(
            [sample["votes"][n] for n in justice_names],
            device=device, dtype=torch.long,
        )

        logits = head(hidden_states, justice_ids)
        preds = logits.argmax(dim=1)

        del hidden_states

        # Per-justice accuracy
        for j, name in enumerate(justice_names):
            correct = (preds[j] == labels[j]).item()
            if name not in all_justice_results:
                all_justice_results[name] = []
            all_justice_results[name].append(correct)

        # Case accuracy
        pred_majority = 0 if (preds == 0).sum() > (preds == 1).sum() else 1
        true_majority = case_result(sample["votes"])
        if true_majority is not None:
            case_results_list.append(pred_majority == true_majority)

    # Summary
    print("\n  Justice vote accuracy:")
    total_correct = 0
    total_counted = 0
    for justice in sorted(all_justice_results):
        results = all_justice_results[justice]
        acc = sum(results) / len(results)
        total_correct += sum(results)
        total_counted += len(results)
        print(f"    {justice:30s}: {acc:5.1%} ({sum(results)}/{len(results)})")

    vote_acc = total_correct / max(total_counted, 1)
    print(f"    {'OVERALL':30s}: {vote_acc:5.1%} ({total_correct}/{total_counted})")

    case_acc = sum(case_results_list) / max(len(case_results_list), 1)
    print(f"\n  Case accuracy: {case_acc:5.1%} "
          f"({sum(case_results_list)}/{len(case_results_list)})")

    # Log layer weights
    layer_weights = head.get_layer_weights()
    print(f"\n  Layer weights (top 5):")
    top_vals, top_idx = layer_weights.topk(5)
    for val, idx in zip(top_vals, top_idx):
        print(f"    Layer {idx.item():2d}: {val.item():.4f}")

    eval_metrics = {
        "eval/vote_acc": vote_acc,
        "eval/case_acc": case_acc,
    }
    for justice in sorted(all_justice_results):
        results = all_justice_results[justice]
        eval_metrics[f"eval/justice/{justice}"] = sum(results) / len(results)
    wandb.log(eval_metrics, step=global_step)

    head.train()


if __name__ == "__main__":
    train()
