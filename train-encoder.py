"""Encoder-based SCOTUS vote predictor with fine-tunable ModernBERT/DeBERTa backbone.

Replaces the frozen LLM approach (train-lee.py) with a smaller, fully fine-tunable
bidirectional encoder. Splits transcripts into petitioner/respondent argument phases,
encodes each with full context, pools to turn-level vectors, and cross-attends from
justice queries with NO residual bypass. Includes auxiliary objective to predict
per-justice question counts.
"""

import argparse
import glob
import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ── Constants ─────────────────────────────────────────────────────────────────

ENCODER_MODELS = {
    "modernbert-base": "answerdotai/ModernBERT-base",
    "modernbert-large": "answerdotai/ModernBERT-large",
    "deberta-v3-base": "microsoft/deberta-v3-base",
    "deberta-v3-large": "microsoft/deberta-v3-large",
}
DEFAULT_ENCODER = "modernbert-base"

DATA_DIR = "case_transcripts_cleaned"
VOTES_DELIMITER = "\n---\nJUSTICE VOTES:\n"
EVAL_YEAR = "2019"
RESULTS_DIR = "results"
MAX_JUSTICES = 128

KNOWN_JUSTICE_NAMES = {
    "White", "Jr.", "Scalia", "Stevens", "Frankfurter", "Rehnquist",
    "Stewart", "Black", "Burger", "Warren", "Marshall", "Breyer",
    "Ginsburg", "Kennedy", "O'Connor", "Souter", "Sotomayor", "Fortas",
    "Blackmun", "Kagan", "Whittaker", "Douglas", "Goldberg", "Clark",
    "Gorsuch", "Reed", "Kavanaugh", "Burton", "Thomas", "Minton", "II",
    "Harlan",
}

PARTY_LABELS = {"Petitioner", "Respondent"}
ALL_SPEAKER_LABELS = KNOWN_JUSTICE_NAMES | PARTY_LABELS | {"Unknown"}

SPEAKER_TYPE_MAP = {"justice": 0, "petitioner": 1, "respondent": 2}


# ── Data types ────────────────────────────────────────────────────────────────


@dataclass
class Turn:
    speaker_label: str   # raw label from transcript
    text: str            # spoken text
    speaker_type: str    # 'justice', 'petitioner', 'respondent', 'unknown'


# ── Data loading (reused from train-lee.py) ───────────────────────────────────


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
            continue

        idx = text.index(VOTES_DELIMITER)
        transcript = text[:idx]
        votes_text = text[idx + len(VOTES_DELIMITER):].strip()

        if not votes_text:
            continue

        votes = parse_votes(votes_text)
        if not votes:
            continue

        samples.append({
            "transcript": transcript,
            "votes": votes,
            "filename": os.path.basename(path),
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
    train_samples = [s for s in samples if not s["filename"].startswith(eval_year)]
    eval_samples = [s for s in samples if s["filename"].startswith(eval_year)]
    print(f"Split: {len(train_samples)} train, {len(eval_samples)} eval (year={eval_year})")
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


# ── Transcript parsing ────────────────────────────────────────────────────────


def classify_speaker(label: str) -> str:
    """Map a speaker label to its type."""
    if label == "Petitioner":
        return "petitioner"
    elif label == "Respondent":
        return "respondent"
    elif label in KNOWN_JUSTICE_NAMES:
        return "justice"
    else:
        return "unknown"


def parse_transcript_turns(transcript: str) -> list[Turn]:
    """Parse speaker-delimited transcript into Turn objects.

    Speaker labels appear as standalone lines. Text on subsequent lines
    until the next speaker label belongs to that speaker.
    """
    lines = transcript.split("\n")
    turns = []
    current_speaker = None
    current_text_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in ALL_SPEAKER_LABELS:
            # Save previous turn
            if current_speaker is not None and current_text_lines:
                text = " ".join(current_text_lines)
                turns.append(Turn(
                    speaker_label=current_speaker,
                    text=text,
                    speaker_type=classify_speaker(current_speaker),
                ))
            current_speaker = stripped
            current_text_lines = []
        else:
            current_text_lines.append(stripped)

    # Last turn
    if current_speaker is not None and current_text_lines:
        text = " ".join(current_text_lines)
        turns.append(Turn(
            speaker_label=current_speaker,
            text=text,
            speaker_type=classify_speaker(current_speaker),
        ))

    return turns


def split_into_phases(turns: list[Turn]) -> tuple[list[Turn], list[Turn]]:
    """Split at first substantial Respondent turn (>=20 words)."""
    for i, turn in enumerate(turns):
        if turn.speaker_type == "respondent" and len(turn.text.split()) >= 20:
            return turns[:i], turns[i:]
    # No substantial respondent turn found — all petitioner phase
    return turns, []


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


# ── Encoder loading ───────────────────────────────────────────────────────────


def freeze_encoder_layers(encoder, n_layers: int):
    """Freeze embedding layer and bottom n_layers of the encoder."""
    # Freeze embeddings
    if hasattr(encoder, "embeddings"):
        for param in encoder.embeddings.parameters():
            param.requires_grad = False

    # Find encoder layers (different models use different attribute names)
    layers = None
    if hasattr(encoder, "layers"):  # ModernBERT
        layers = encoder.layers
    elif hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):  # DeBERTa
        layers = encoder.encoder.layer

    if layers is not None:
        for i in range(min(n_layers, len(layers))):
            for param in layers[i].parameters():
                param.requires_grad = False


def load_encoder(name: str, freeze_layers: int = 0):
    """Load encoder model and tokenizer.

    Returns (encoder, tokenizer, hidden_dim, max_length).
    """
    model_path = ENCODER_MODELS[name]
    print(f"Loading encoder: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoder = AutoModel.from_pretrained(model_path)

    hidden_dim = encoder.config.hidden_size
    max_length = getattr(encoder.config, "max_position_embeddings", 512)

    # Enable gradient checkpointing to save memory
    if hasattr(encoder, "gradient_checkpointing_enable"):
        encoder.gradient_checkpointing_enable()

    if freeze_layers > 0:
        freeze_encoder_layers(encoder, freeze_layers)
        n_frozen = sum(1 for p in encoder.parameters() if not p.requires_grad)
        n_total = sum(1 for p in encoder.parameters())
        print(f"Froze {n_frozen}/{n_total} encoder parameters "
              f"(bottom {freeze_layers} layers + embeddings)")

    n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Encoder: hidden_dim={hidden_dim}, max_length={max_length}, "
          f"trainable params={n_trainable:,}")

    return encoder, tokenizer, hidden_dim, max_length


# ── Phase encoding ────────────────────────────────────────────────────────────


def encode_phase_with_turn_pooling(
    encoder: nn.Module,
    tokenizer,
    turns: list[Turn],
    max_length: int,
    device: torch.device,
    amp_ctx,
) -> torch.Tensor | None:
    """Encode a phase's turns with full context, then pool per-turn.

    Returns (T, hidden_dim) turn vectors, or None if no turns.
    """
    if not turns:
        return None

    # Build text with tracked character boundaries per turn
    segments = []
    char_starts = []
    char_ends = []
    pos = 0
    for turn in turns:
        char_starts.append(pos)
        segments.append(turn.text)
        pos += len(turn.text)
        char_ends.append(pos)
        # Add space separator between turns
        segments.append(" ")
        pos += 1

    full_text = "".join(segments)

    # Tokenize with offset mapping
    encoding = tokenizer(
        full_text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=True,
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offset_mapping = encoding["offset_mapping"][0]  # (seq_len, 2), stays on CPU

    # Forward through encoder
    with amp_ctx:
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
    hidden_states = outputs.last_hidden_state[0].float()  # (seq_len, hidden_dim)

    # Identify special tokens (offset (0,0) means special token)
    is_special = (offset_mapping[:, 0] == 0) & (offset_mapping[:, 1] == 0)

    # Map tokens to turns and mean-pool
    turn_vectors = []
    token_starts = offset_mapping[:, 0]

    for t_idx in range(len(turns)):
        cs, ce = char_starts[t_idx], char_ends[t_idx]
        mask = (token_starts >= cs) & (token_starts < ce) & ~is_special

        if mask.any():
            turn_vectors.append(hidden_states[mask.to(device)].mean(dim=0))
        else:
            turn_vectors.append(torch.zeros(hidden_states.shape[-1], device=device))

    return torch.stack(turn_vectors)  # (T, hidden_dim)


# ── Model components ──────────────────────────────────────────────────────────


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
        base = justice_ids.unsqueeze(1) * self.num_queries  # (J, 1)
        offsets = torch.arange(self.num_queries, device=justice_ids.device)  # (Q,)
        indices = base + offsets  # (J, Q)
        return self.embedding(indices)  # (J, Q, dim)


class TranscriptCrossAttention(nn.Module):
    """Cross-attention from justice queries to transcript turn vectors.

    CRITICAL: No residual connection from justice queries. Output comes entirely
    from attending to transcript content. Without transcript, all justices get
    identical representations (LayerNorm bias), forcing the model to use transcript.
    """

    def __init__(self, input_dim: int, head_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.kv_proj = nn.Linear(input_dim, inner_dim * 2, bias=False)
        self.out_proj = nn.Linear(inner_dim, inner_dim, bias=False)
        self.layer_norm = nn.LayerNorm(inner_dim)
        self.attn_dropout = dropout

    def forward(
        self, justice_queries: torch.Tensor, transcript: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            justice_queries: (J, num_heads, head_dim)
            transcript: (T, input_dim) turn vectors
        Returns:
            output: (J, num_heads, head_dim) — NO residual from queries
        """
        J, H, D = justice_queries.shape
        T = transcript.shape[0]

        # Project transcript turns to K, V
        kv = self.kv_proj(transcript)  # (T, H*D*2)
        k, v = kv.chunk(2, dim=-1)  # each (T, H*D)
        k = k.view(T, H, D).permute(1, 0, 2)  # (H, T, D)
        v = v.view(T, H, D).permute(1, 0, 2)  # (H, T, D)

        # Expand K, V for all justices: (H, T, D) -> (J*H, T, D)
        k = k.unsqueeze(0).expand(J, -1, -1, -1).reshape(J * H, T, D)
        v = v.unsqueeze(0).expand(J, -1, -1, -1).reshape(J * H, T, D)
        q = justice_queries.reshape(J * H, 1, D)  # (J*H, 1, D)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )  # (J*H, 1, D)
        attn_out = attn_out.view(J, H, D)

        # Output projection + LayerNorm (NO residual from justice_queries)
        flat = attn_out.view(J, H * D)  # (J, H*D)
        out = self.out_proj(flat)  # (J, H*D)
        out = self.layer_norm(out).view(J, H, D)  # (J, H, D)
        return out


class AuxQuestionCountHead(nn.Module):
    """Predicts per-justice question counts per phase from cross-attention output."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (J, input_dim) — flattened cross-attention output per justice
        Returns:
            predictions: (J, 2) — predicted (pet_count, resp_count) normalized
        """
        return self.mlp(x)


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


class SCOTUSEncoderModel(nn.Module):
    """Full model: justice embeddings + cross-attention + self-attention + classifiers.

    Takes pre-computed turn vectors (from encoder) and justice IDs.
    """

    def __init__(
        self,
        encoder_dim: int,
        head_dim: int = 128,
        num_queries: int = 4,
        self_attn_layers: int = 1,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        max_justices: int = MAX_JUSTICES,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.head_dim = head_dim
        inner_dim = num_queries * head_dim

        # Speaker type embedding (added to turn vectors before projection)
        self.speaker_type_emb = nn.Embedding(3, encoder_dim)

        # Project encoder dim to cross-attention input dim
        self.input_proj = nn.Linear(encoder_dim, inner_dim)

        # Justice query embeddings
        self.justice_embeddings = JusticeEmbeddings(max_justices, num_queries, head_dim)

        # Cross-attention (NO residual)
        self.cross_attention = TranscriptCrossAttention(
            inner_dim, head_dim, num_queries, dropout=dropout
        )

        # Self-attention across justices
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=inner_dim,
            nhead=num_queries,
            dim_feedforward=ffn_dim,
            batch_first=True,
            norm_first=True,
            dropout=dropout,
        )
        self.self_attention = nn.TransformerEncoder(
            encoder_layer, num_layers=self_attn_layers
        )

        # Vote classifier
        self.vote_classifier = VoteClassifier(num_queries, head_dim)

        # Auxiliary question count head
        self.aux_head = AuxQuestionCountHead(inner_dim)

    def forward(
        self,
        turn_vectors: torch.Tensor,
        turn_speaker_types: torch.Tensor,
        justice_ids: torch.Tensor,
        no_transcript: bool = False,
        no_speaker_embeddings: bool = False,
        no_self_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            turn_vectors: (T, encoder_dim) pooled turn vectors from encoder
            turn_speaker_types: (T,) speaker type IDs (0=justice, 1=pet, 2=resp)
            justice_ids: (J,) integer justice IDs
            no_transcript: zero out transcript for ablation
            no_speaker_embeddings: skip speaker type embeddings
            no_self_attention: skip self-attention across justices
        Returns:
            logits: (J, 2) vote logits
            aux_preds: (J, 2) auxiliary question count predictions
        """
        # Add speaker type embeddings
        if not no_speaker_embeddings:
            turn_vectors = turn_vectors + self.speaker_type_emb(turn_speaker_types)

        # Zero out for ablation
        if no_transcript:
            turn_vectors = torch.zeros_like(turn_vectors)

        # Project to cross-attention input dimension
        transcript = self.input_proj(turn_vectors)  # (T, inner_dim)

        # Justice queries
        queries = self.justice_embeddings(justice_ids)  # (J, Q, head_dim)

        # Cross-attend to transcript (NO residual)
        cross_out = self.cross_attention(queries, transcript)  # (J, Q, head_dim)

        # Auxiliary head (from flattened cross-attention output)
        J = cross_out.shape[0]
        cross_flat = cross_out.view(J, -1)  # (J, Q*head_dim)
        aux_preds = self.aux_head(cross_flat)  # (J, 2)

        # Self-attention across justices
        if no_self_attention:
            self_out = cross_out
        else:
            flat = cross_out.view(J, -1).unsqueeze(0)  # (1, J, Q*head_dim)
            self_out = self.self_attention(flat)  # (1, J, Q*head_dim)
            self_out = self_out.squeeze(0).view(J, self.num_queries, self.head_dim)

        # Classify votes
        logits = self.vote_classifier(self_out)  # (J, 2)
        return logits, aux_preds


# ── Auxiliary objective helpers ────────────────────────────────────────────────


def compute_aux_targets(
    justice_names: list[str],
    pet_turns: list[Turn],
    resp_turns: list[Turn],
    device: torch.device,
) -> torch.Tensor:
    """Compute normalized question count targets per justice.

    Returns (J, 2) tensor with (pet_count, resp_count) normalized by
    the max count across all justices and phases in this case.
    """
    J = len(justice_names)
    counts = torch.zeros(J, 2)

    for j, name in enumerate(justice_names):
        for turn in pet_turns:
            if turn.speaker_label == name:
                counts[j, 0] += 1
        for turn in resp_turns:
            if turn.speaker_label == name:
                counts[j, 1] += 1

    max_count = counts.max().item()
    if max_count > 0:
        counts = counts / max_count

    return counts.to(device)


# ── Results saving ────────────────────────────────────────────────────────────


def save_results(
    run_config: dict,
    epochs: list[dict],
    results_dir: str,
    run_id: str | None = None,
) -> str:
    """Write run config and per-epoch stats to results_dir (JSON + CSV)."""
    os.makedirs(results_dir, exist_ok=True)
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(results_dir, f"run_{run_id}.json")
    payload = {
        "run_id": run_id,
        "config": run_config,
        "epochs": epochs,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    csv_path = os.path.join(results_dir, f"metrics_{run_id}.csv")
    csv_keys = [
        "epoch", "train_loss", "train_vote_acc", "train_case_acc",
        "eval_vote_acc", "eval_case_acc", "eval_est_vote_acc", "eval_est_case_acc",
    ]
    if epochs:
        with open(csv_path, "w") as f:
            f.write(",".join(csv_keys) + "\n")
            for row in epochs:
                f.write(",".join(str(row.get(k, "")) for k in csv_keys) + "\n")

    print(f"Results saved: {json_path}, {csv_path}")
    return json_path


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Encoder-based SCOTUS vote predictor"
    )

    # Encoder
    p.add_argument(
        "--encoder", choices=list(ENCODER_MODELS.keys()), default=DEFAULT_ENCODER,
        help=f"Encoder model (default: {DEFAULT_ENCODER})",
    )
    p.add_argument(
        "--max-length", type=int, default=0,
        help="Override encoder max seq length (0=model default)",
    )

    # Training
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-5, help="Encoder learning rate")
    p.add_argument("--head-lr", type=float, default=0, help="Head LR (0 → lr*10)")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)

    # Head architecture
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--num-queries", type=int, default=4)
    p.add_argument("--self-attn-layers", type=int, default=1)
    p.add_argument("--ffn-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)

    # Encoder control
    p.add_argument(
        "--freeze-layers", type=int, default=0,
        help="Freeze bottom N encoder layers + embeddings",
    )

    # Ablations
    p.add_argument("--no-transcript", action="store_true",
                    help="Zero out transcript (critical diagnostic)")
    p.add_argument("--no-speaker-embeddings", action="store_true",
                    help="Disable speaker-type embeddings")
    p.add_argument("--no-self-attention", action="store_true",
                    help="Disable self-attention across justices")
    p.add_argument("--aux-weight", type=float, default=0.1,
                    help="Auxiliary loss weight (0=disable)")

    # Output
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--eval-year", default=EVAL_YEAR)
    p.add_argument("--results-dir", default=RESULTS_DIR)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--log-every", type=int, default=8,
                    help="Log metrics every N optimizer steps")

    return p.parse_args()


# ── Prepare sample ────────────────────────────────────────────────────────────


def prepare_sample(
    sample: dict,
    encoder: nn.Module,
    tokenizer,
    max_length: int,
    device: torch.device,
    amp_ctx,
    no_transcript: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[Turn], list[Turn]]:
    """Parse transcript, encode phases, return turn vectors and metadata.

    Returns:
        turn_vectors: (T, encoder_dim) — concatenated turn vectors from both phases
        turn_speaker_types: (T,) — speaker type IDs
        pet_turns: list of Turn objects for petitioner phase
        resp_turns: list of Turn objects for respondent phase
    """
    turns = parse_transcript_turns(sample["transcript"])
    pet_turns, resp_turns = split_into_phases(turns)

    if no_transcript:
        # Single zero vector for ablation
        hidden_dim = encoder.config.hidden_size
        turn_vectors = torch.zeros(1, hidden_dim, device=device)
        turn_speaker_types = torch.tensor([0], device=device)
        return turn_vectors, turn_speaker_types, pet_turns, resp_turns

    all_turn_vectors = []
    all_speaker_types = []

    for phase_turns in [pet_turns, resp_turns]:
        if not phase_turns:
            continue
        vectors = encode_phase_with_turn_pooling(
            encoder, tokenizer, phase_turns, max_length, device, amp_ctx,
        )
        if vectors is not None:
            all_turn_vectors.append(vectors)
            for turn in phase_turns:
                all_speaker_types.append(
                    SPEAKER_TYPE_MAP.get(turn.speaker_type, 0)
                )

    if not all_turn_vectors:
        # Fallback: no parseable turns
        hidden_dim = encoder.config.hidden_size
        turn_vectors = torch.zeros(1, hidden_dim, device=device)
        turn_speaker_types = torch.tensor([0], device=device)
    else:
        turn_vectors = torch.cat(all_turn_vectors, dim=0)
        turn_speaker_types = torch.tensor(all_speaker_types, device=device)

    return turn_vectors, turn_speaker_types, pet_turns, resp_turns


# ── Training ──────────────────────────────────────────────────────────────────


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or f"encoder-{args.encoder}"
    head_lr = args.head_lr if args.head_lr > 0 else args.lr * 10

    # AMP context for encoder forward passes
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    os.environ.setdefault("WANDB_PROJECT", "supreme-court-encoder")
    config = {
        "encoder": args.encoder,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "lr": args.lr,
        "head_lr": head_lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "grad_accum": args.grad_accum,
        "grad_clip": args.grad_clip,
        "label_smoothing": args.label_smoothing,
        "head_dim": args.head_dim,
        "num_queries": args.num_queries,
        "self_attn_layers": args.self_attn_layers,
        "ffn_dim": args.ffn_dim,
        "dropout": args.dropout,
        "freeze_layers": args.freeze_layers,
        "no_transcript": args.no_transcript,
        "no_speaker_embeddings": args.no_speaker_embeddings,
        "no_self_attention": args.no_self_attention,
        "aux_weight": args.aux_weight,
        "eval_year": args.eval_year,
    }
    wandb.init(project="supreme-court-encoder", name=run_name, config=config)

    # Load data
    all_samples = load_transcripts(args.data_dir)
    train_samples, eval_samples = split_by_year(all_samples, args.eval_year)

    registry = JusticeRegistry()
    registry.build_from_samples(all_samples)

    # Load encoder
    encoder, tokenizer, encoder_dim, model_max_length = load_encoder(
        args.encoder, freeze_layers=args.freeze_layers
    )
    max_length = args.max_length if args.max_length > 0 else model_max_length
    encoder.to(device)

    # Build head
    model = SCOTUSEncoderModel(
        encoder_dim=encoder_dim,
        head_dim=args.head_dim,
        num_queries=args.num_queries,
        self_attn_layers=args.self_attn_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
    ).to(device).float()

    total_head_params = sum(p.numel() for p in model.parameters())
    total_enc_trainable = sum(
        p.numel() for p in encoder.parameters() if p.requires_grad
    )
    print(f"Head parameters: {total_head_params:,}")
    print(f"Encoder trainable parameters: {total_enc_trainable:,}")
    print(f"Total trainable: {total_head_params + total_enc_trainable:,}")

    # Optimizer with separate LR groups
    optimizer = torch.optim.AdamW([
        {"params": [p for p in encoder.parameters() if p.requires_grad],
         "lr": args.lr},
        {"params": model.parameters(), "lr": head_lr},
    ], weight_decay=args.weight_decay)

    # Cosine schedule with linear warmup
    total_steps = (len(train_samples) * args.epochs) // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(1e-3, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Print parse verification for a few cases
    print("\n── Turn parsing verification ──")
    for sample in train_samples[:3]:
        turns = parse_transcript_turns(sample["transcript"])
        pet, resp = split_into_phases(turns)
        justice_turns = [t for t in turns if t.speaker_type == "justice"]
        print(f"  {sample['filename']}: {len(turns)} turns total, "
              f"{len(pet)} petitioner phase, {len(resp)} respondent phase, "
              f"{len(justice_turns)} justice turns")
    print()

    # Training loop
    global_step = 0
    accum_loss = 0.0
    accum_aux_loss = 0.0
    accum_correct = 0
    accum_total = 0
    accum_case_correct = 0
    accum_cases = 0
    accum_vote_prob_sum = 0.0
    accum_vote_prob_n = 0
    accum_case_prob_sum = 0.0
    accum_case_prob_n = 0
    log_steps_accum = 0
    log_micro_batches = 0
    micro_step = 0
    epochs_stats = []

    for epoch in range(args.epochs):
        encoder.train()
        model.train()

        # Shuffle training data
        indices = list(range(len(train_samples)))
        random.shuffle(indices)

        pbar = tqdm(range(len(train_samples)), desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx in pbar:
            sample = train_samples[indices[batch_idx]]

            # Encode transcript
            turn_vectors, turn_speaker_types, pet_turns, resp_turns = prepare_sample(
                sample, encoder, tokenizer, max_length, device, amp_ctx,
                no_transcript=args.no_transcript,
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
            logits, aux_preds = model(
                turn_vectors, turn_speaker_types, justice_ids,
                no_transcript=args.no_transcript,
                no_speaker_embeddings=args.no_speaker_embeddings,
                no_self_attention=args.no_self_attention,
            )

            # Vote loss
            vote_loss = F.cross_entropy(
                logits, labels, label_smoothing=args.label_smoothing,
            )

            # Auxiliary loss
            aux_loss = torch.tensor(0.0, device=device)
            if args.aux_weight > 0:
                aux_targets = compute_aux_targets(
                    justice_names, pet_turns, resp_turns, device
                )
                aux_loss = F.mse_loss(aux_preds, aux_targets)

            total_loss = (vote_loss + args.aux_weight * aux_loss) / args.grad_accum
            total_loss.backward()

            # Track metrics (detached)
            with torch.no_grad():
                accum_loss += vote_loss.item()
                accum_aux_loss += aux_loss.item()
                preds = logits.argmax(dim=1)
                accum_correct += (preds == labels).sum().item()
                accum_total += len(labels)

                probs = logits.softmax(dim=1)
                correct_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                accum_vote_prob_sum += correct_probs.sum().item()
                accum_vote_prob_n += len(labels)

                pred_majority = 0 if (preds == 0).sum() > (preds == 1).sum() else 1
                true_majority = case_result(sample["votes"])
                if true_majority is not None:
                    accum_case_correct += int(pred_majority == true_majority)
                    accum_cases += 1
                    accum_case_prob_sum += majority_correct_prob(
                        correct_probs.tolist()
                    )
                    accum_case_prob_n += 1

            del turn_vectors, turn_speaker_types, logits, aux_preds, total_loss
            log_micro_batches += 1
            micro_step += 1

            # Optimizer step
            if micro_step % args.grad_accum == 0 or (batch_idx + 1) == len(train_samples):
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        list(encoder.parameters()) + list(model.parameters()),
                        args.grad_clip,
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                log_steps_accum += 1

                # Log metrics
                if log_steps_accum >= args.log_every or (batch_idx + 1) == len(train_samples):
                    n = max(log_micro_batches, 1)
                    metrics = {
                        "train/loss": accum_loss / n,
                        "train/aux_loss": accum_aux_loss / n,
                        "train/greedy_vote_acc": accum_correct / max(accum_total, 1),
                        "train/greedy_case_acc": accum_case_correct / max(accum_cases, 1),
                        "train/est_vote_acc": accum_vote_prob_sum / max(accum_vote_prob_n, 1),
                        "train/est_case_acc": accum_case_prob_sum / max(accum_case_prob_n, 1),
                        "train/lr_encoder": scheduler.get_last_lr()[0],
                        "train/lr_head": scheduler.get_last_lr()[1],
                        "train/step": global_step,
                    }
                    wandb.log(metrics, step=global_step)
                    pbar.set_postfix(
                        loss=f"{accum_loss / n:.4f}",
                        gv=f"{accum_correct / max(accum_total, 1):.3f}",
                        ev=f"{accum_vote_prob_sum / max(accum_vote_prob_n, 1):.3f}",
                        gc=f"{accum_case_correct / max(accum_cases, 1):.3f}",
                        ec=f"{accum_case_prob_sum / max(accum_case_prob_n, 1):.3f}",
                    )

                    accum_loss = 0.0
                    accum_aux_loss = 0.0
                    accum_correct = 0
                    accum_total = 0
                    accum_case_correct = 0
                    accum_cases = 0
                    accum_vote_prob_sum = 0.0
                    accum_vote_prob_n = 0
                    accum_case_prob_sum = 0.0
                    accum_case_prob_n = 0
                    log_steps_accum = 0
                    log_micro_batches = 0

                micro_step = 0

        # End of epoch evaluation
        epoch_stats = {"epoch": epoch + 1}
        if eval_samples:
            eval_metrics = evaluate(
                model, encoder, tokenizer, eval_samples, registry,
                device, global_step, epoch, args, max_length, amp_ctx,
            )
            epoch_stats.update(eval_metrics)
        epochs_stats.append(epoch_stats)

    # Save results
    run_id = run_name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(config, epochs_stats, args.results_dir, run_id=run_id)

    # Save model weights
    save_dir = "output"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{run_name}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "registry": registry.name_to_id,
        "config": {
            "encoder": args.encoder,
            "encoder_dim": encoder_dim,
            "head_dim": args.head_dim,
            "num_queries": args.num_queries,
            "self_attn_layers": args.self_attn_layers,
            "ffn_dim": args.ffn_dim,
            "dropout": args.dropout,
            "max_justices": MAX_JUSTICES,
        },
    }, save_path)
    print(f"Saved model to {save_path}")

    wandb.finish()


# ── Evaluation ────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(
    model: SCOTUSEncoderModel,
    encoder: nn.Module,
    tokenizer,
    eval_samples: list[dict],
    registry: JusticeRegistry,
    device: torch.device,
    global_step: int,
    epoch: int,
    args,
    max_length: int,
    amp_ctx,
) -> dict:
    """Evaluate on held-out samples. Returns dict with eval metrics."""
    encoder.eval()
    model.eval()

    all_justice_results: dict[str, list[bool]] = {}
    all_justice_probs: dict[str, list[float]] = {}
    case_results_list: list[bool] = []
    case_prob_list: list[float] = []

    print(f"\n── Evaluation ({len(eval_samples)} cases, epoch {epoch+1}) ──\n")

    for sample in tqdm(eval_samples, desc="Evaluating"):
        turn_vectors, turn_speaker_types, pet_turns, resp_turns = prepare_sample(
            sample, encoder, tokenizer, max_length, device, amp_ctx,
            no_transcript=args.no_transcript,
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

        logits, _ = model(
            turn_vectors, turn_speaker_types, justice_ids,
            no_transcript=args.no_transcript,
            no_speaker_embeddings=args.no_speaker_embeddings,
            no_self_attention=args.no_self_attention,
        )

        preds = logits.argmax(dim=1)
        probs = logits.softmax(dim=1)
        correct_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)

        for j, name in enumerate(justice_names):
            correct = (preds[j] == labels[j]).item()
            if name not in all_justice_results:
                all_justice_results[name] = []
                all_justice_probs[name] = []
            all_justice_results[name].append(correct)
            all_justice_probs[name].append(correct_probs[j].item())

        pred_majority = 0 if (preds == 0).sum() > (preds == 1).sum() else 1
        true_majority = case_result(sample["votes"])
        if true_majority is not None:
            case_results_list.append(pred_majority == true_majority)
            case_prob_list.append(majority_correct_prob(correct_probs.tolist()))

        del turn_vectors, turn_speaker_types

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

    eval_metrics = {
        "eval/greedy_vote_acc": greedy_vote_acc,
        "eval/greedy_case_acc": greedy_case_acc,
        "eval/est_vote_acc": est_vote_acc,
        "eval/est_case_acc": est_case_acc,
    }
    for justice in sorted(all_justice_results):
        results = all_justice_results[justice]
        probs_list = all_justice_probs[justice]
        eval_metrics[f"eval/justice_greedy/{justice}"] = sum(results) / len(results)
        eval_metrics[f"eval/justice_est/{justice}"] = sum(probs_list) / len(probs_list)
    wandb.log(eval_metrics, step=global_step)

    encoder.train()
    model.train()

    return {
        "eval_vote_acc": greedy_vote_acc,
        "eval_case_acc": greedy_case_acc,
        "eval_est_vote_acc": est_vote_acc,
        "eval_est_case_acc": est_case_acc,
    }


if __name__ == "__main__":
    train()
