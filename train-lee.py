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
DATA_DIR = "case_transcripts_cleaned"
VOTES_DELIMITER = "\n---\nJUSTICE VOTES:\n"
EVAL_YEAR = "2019"

# Training
BATCH_SIZE = 1
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
GRAD_ACCUM_STEPS = 16  # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS = 16
WARMUP_RATIO = 0.05
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


def majority_correct_prob(probs_correct: list[float]) -> float:
    """P(majority of votes correct) via Poisson binomial DP."""
    n = len(probs_correct)
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

    def weights(self) -> torch.Tensor:
        """Return softmax-normalized layer weights."""
        return F.softmax(self.layer_logits, dim=0)


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
            return self.layer_weighter.weights()


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


def get_blended_hidden_states(
    backbone,
    tokenizer,
    texts: list[str],
    max_length: int,
    device: torch.device,
    layer_weighter: LayerWeighter,
) -> list[torch.Tensor]:
    """Run frozen backbone on a batch, blend hidden states incrementally.

    Tokenizes and pads a batch of texts, runs one batched backbone forward,
    then blends layers one at a time to minimize memory. Returns a list of
    per-sample blended tensors (unpadded to original lengths).

    Peak hidden-state overhead: ~2 × (B, S_max, D) in float32 instead of 29×.
    """
    # Tokenize each text, left-truncate
    all_ids = []
    lengths = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > max_length:
            ids = ids[-max_length:]
        all_ids.append(ids)
        lengths.append(len(ids))

    # Pad to max length in batch (right-pad)
    max_len = max(lengths)
    pad_id = tokenizer.pad_token_id
    padded = [ids + [pad_id] * (max_len - len(ids)) for ids in all_ids]
    input_tensor = torch.tensor(padded, device=device)  # (B, S_max)
    attention_mask = torch.tensor(
        [[1] * l + [0] * (max_len - l) for l in lengths],
        device=device,
    )  # (B, S_max)

    with torch.no_grad():
        outputs = backbone(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    # Blend incrementally: one layer at a time
    weights = layer_weighter.weights()  # (num_layers,)
    blended = None
    for i, h in enumerate(outputs.hidden_states):
        weighted = weights[i] * h.detach().float()  # (B, S_max, D)
        if blended is None:
            blended = weighted
        else:
            blended += weighted
        del weighted

    del outputs, input_tensor, attention_mask
    torch.cuda.empty_cache()

    # Unpad: return list of (1, S_i, D) tensors per sample
    result = []
    for b, l in enumerate(lengths):
        result.append(blended[b:b+1, :l, :])  # (1, S_i, D)

    return result


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
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM_STEPS,
            "effective_batch": BATCH_SIZE * GRAD_ACCUM_STEPS,
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

    # Optimizer and scheduler (linear warmup + cosine decay)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY
    )
    effective_batch = BATCH_SIZE * GRAD_ACCUM_STEPS
    total_steps = (len(train_samples) * args.epochs) // effective_batch
    warmup_steps = int(total_steps * WARMUP_RATIO)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(1e-6 / args.lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    global_step = 0
    accum_loss = 0.0
    accum_correct = 0
    accum_total = 0
    accum_case_correct = 0
    accum_cases = 0
    accum_vote_prob_sum = 0.0  # estimated: sum of P(correct vote)
    accum_vote_prob_n = 0
    accum_case_prob_sum = 0.0  # estimated: sum of P(majority correct)
    accum_case_prob_n = 0
    micro_step = 0  # counts batches within a grad accum window

    for epoch in range(args.epochs):
        head.train()
        num_batches = (len(train_samples) + BATCH_SIZE - 1) // BATCH_SIZE
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx in pbar:
            batch_start = batch_idx * BATCH_SIZE
            batch_samples = train_samples[batch_start : batch_start + BATCH_SIZE]

            # Batched backbone forward + incremental blend
            blended_list = get_blended_hidden_states(
                backbone, tokenizer,
                [s["transcript"] for s in batch_samples],
                MAX_SEQ_LENGTH, device, head.layer_weighter,
            )

            # Process each sample in the batch through the head
            batch_loss = 0.0
            accumulated_loss = None
            sample_logits_labels = []
            for sample, blended in zip(batch_samples, blended_list):
                justice_names = list(sample["votes"].keys())
                justice_ids = torch.tensor(
                    [registry.get_or_add(n) for n in justice_names],
                    device=device,
                )
                labels = torch.tensor(
                    [sample["votes"][n] for n in justice_names],
                    device=device, dtype=torch.long,
                )

                logits = head(blended, justice_ids)  # (J, 2)
                loss = F.cross_entropy(logits, labels) / (GRAD_ACCUM_STEPS * len(batch_samples))
                accumulated_loss = loss if accumulated_loss is None else accumulated_loss + loss
                batch_loss += loss.item() * GRAD_ACCUM_STEPS * len(batch_samples)
                sample_logits_labels.append((logits.detach(), labels, sample))

            accumulated_loss.backward()
            del accumulated_loss

            # Compute metrics (no grad needed, logits already detached)
            for logits, labels, sample in sample_logits_labels:
                preds = logits.argmax(dim=1)
                accum_correct += (preds == labels).sum().item()
                accum_total += len(labels)

                # Estimated accuracy: P(correct) per vote from softmax
                probs = logits.softmax(dim=1)  # (J, 2)
                correct_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                accum_vote_prob_sum += correct_probs.sum().item()
                accum_vote_prob_n += len(labels)

                # Case level
                pred_majority = 0 if (preds == 0).sum() > (preds == 1).sum() else 1
                true_majority = case_result(sample["votes"])
                if true_majority is not None:
                    accum_case_correct += int(pred_majority == true_majority)
                    accum_cases += 1
                    accum_case_prob_sum += majority_correct_prob(correct_probs.tolist())
                    accum_case_prob_n += 1
            del sample_logits_labels

            del blended_list
            accum_loss += batch_loss / len(batch_samples)
            micro_step += 1

            # Optimizer step after GRAD_ACCUM_STEPS micro-batches
            if micro_step % GRAD_ACCUM_STEPS == 0 or (batch_idx + 1) == num_batches:
                nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                layer_weights = head.get_layer_weights()
                num_accum = micro_step if micro_step <= GRAD_ACCUM_STEPS else GRAD_ACCUM_STEPS
                metrics = {
                    "train/loss": accum_loss / num_accum,
                    "train/greedy_vote_acc": accum_correct / max(accum_total, 1),
                    "train/greedy_case_acc": accum_case_correct / max(accum_cases, 1),
                    "train/est_vote_acc": accum_vote_prob_sum / max(accum_vote_prob_n, 1),
                    "train/est_case_acc": accum_case_prob_sum / max(accum_case_prob_n, 1),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/step": global_step,
                }
                top_vals, top_idx = layer_weights.topk(5)
                for rank, (val, idx) in enumerate(zip(top_vals, top_idx)):
                    metrics[f"layers/top{rank+1}_layer"] = idx.item()
                    metrics[f"layers/top{rank+1}_weight"] = val.item()

                wandb.log(metrics, step=global_step)
                pbar.set_postfix(
                    loss=f"{accum_loss / num_accum:.4f}",
                    gv=f"{accum_correct / max(accum_total, 1):.3f}",
                    ev=f"{accum_vote_prob_sum / max(accum_vote_prob_n, 1):.3f}",
                    gc=f"{accum_case_correct / max(accum_cases, 1):.3f}",
                    ec=f"{accum_case_prob_sum / max(accum_case_prob_n, 1):.3f}",
                )

                accum_loss = 0.0
                accum_correct = 0
                accum_total = 0
                accum_case_correct = 0
                accum_cases = 0
                accum_vote_prob_sum = 0.0
                accum_vote_prob_n = 0
                accum_case_prob_sum = 0.0
                accum_case_prob_n = 0
                micro_step = 0

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
    all_justice_probs: dict[str, list[float]] = {}  # estimated P(correct) per justice
    case_results_list: list[bool] = []
    case_prob_list: list[float] = []  # estimated P(majority correct) per case

    print(f"\n── Evaluation ({len(eval_samples)} cases, epoch {epoch+1}) ──\n")

    num_batches = (len(eval_samples) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
        batch_start = batch_idx * BATCH_SIZE
        batch_samples = eval_samples[batch_start : batch_start + BATCH_SIZE]

        blended_list = get_blended_hidden_states(
            backbone, tokenizer,
            [s["transcript"] for s in batch_samples],
            MAX_SEQ_LENGTH, device, head.layer_weighter,
        )

        for sample, blended in zip(batch_samples, blended_list):
            justice_names = list(sample["votes"].keys())
            justice_ids = torch.tensor(
                [registry.get_or_add(n) for n in justice_names],
                device=device,
            )
            labels = torch.tensor(
                [sample["votes"][n] for n in justice_names],
                device=device, dtype=torch.long,
            )

            logits = head(blended, justice_ids)
            preds = logits.argmax(dim=1)
            probs = logits.softmax(dim=1)  # (J, 2)
            correct_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)

            # Per-justice accuracy (greedy + estimated)
            for j, name in enumerate(justice_names):
                correct = (preds[j] == labels[j]).item()
                if name not in all_justice_results:
                    all_justice_results[name] = []
                    all_justice_probs[name] = []
                all_justice_results[name].append(correct)
                all_justice_probs[name].append(correct_probs[j].item())

            # Case accuracy (greedy + estimated)
            pred_majority = 0 if (preds == 0).sum() > (preds == 1).sum() else 1
            true_majority = case_result(sample["votes"])
            if true_majority is not None:
                case_results_list.append(pred_majority == true_majority)
                case_prob_list.append(majority_correct_prob(correct_probs.tolist()))

        del blended_list

    # Summary
    print("\n  Justice vote accuracy (greedy / estimated):")
    total_correct = 0
    total_counted = 0
    total_prob_sum = 0.0
    for justice in sorted(all_justice_results):
        results = all_justice_results[justice]
        probs = all_justice_probs[justice]
        greedy = sum(results) / len(results)
        est = sum(probs) / len(probs)
        total_correct += sum(results)
        total_counted += len(results)
        total_prob_sum += sum(probs)
        print(f"    {justice:30s}: {greedy:5.1%} / {est:5.1%}  ({len(results)} votes)")

    greedy_vote_acc = total_correct / max(total_counted, 1)
    est_vote_acc = total_prob_sum / max(total_counted, 1)
    print(f"    {'OVERALL':30s}: {greedy_vote_acc:5.1%} / {est_vote_acc:5.1%}")

    greedy_case_acc = sum(case_results_list) / max(len(case_results_list), 1)
    est_case_acc = sum(case_prob_list) / max(len(case_prob_list), 1)
    print(f"\n  Case accuracy (greedy / estimated): "
          f"{greedy_case_acc:5.1%} / {est_case_acc:5.1%}  "
          f"({len(case_results_list)} cases)")

    # Log layer weights
    layer_weights = head.get_layer_weights()
    print(f"\n  Layer weights (top 5):")
    top_vals, top_idx = layer_weights.topk(5)
    for val, idx in zip(top_vals, top_idx):
        print(f"    Layer {idx.item():2d}: {val.item():.4f}")

    eval_metrics = {
        "eval/greedy_vote_acc": greedy_vote_acc,
        "eval/greedy_case_acc": greedy_case_acc,
        "eval/est_vote_acc": est_vote_acc,
        "eval/est_case_acc": est_case_acc,
    }
    for justice in sorted(all_justice_results):
        results = all_justice_results[justice]
        probs = all_justice_probs[justice]
        eval_metrics[f"eval/justice_greedy/{justice}"] = sum(results) / len(results)
        eval_metrics[f"eval/justice_est/{justice}"] = sum(probs) / len(probs)
    wandb.log(eval_metrics, step=global_step)

    head.train()


if __name__ == "__main__":
    train()
