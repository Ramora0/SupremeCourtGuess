"""
SCOTUS vote prediction (v3): hierarchical chunking + train-lee-style head.

- Chunking: same as before — global_start, middle chunks, global_end; each chunk
  encoded separately; no single long sequence.
- Architecture: same as train-lee — layer blending (ELMo-style), justice embeddings
  (variable justices, 8 queries each), cross-attention to chunk embeddings (with
  chunk mask), self-attention over justice tokens, LayerNorm+GELU+Linear classifier.
- Vote convention: 0 = Petitioner, 1 = Respondent (train-lee).
"""

import argparse
import glob
import json
import math
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = "case_transcripts_cleaned"
VOTES_DELIMITER = "\n---\nJUSTICE VOTES:\n"

MODELS = {
    "7b": "Qwen/Qwen2.5-7B",
    "3b": "Qwen/Qwen2.5-3B",
    "1.5b": "Qwen/Qwen2.5-1.5B",
    "0.5b": "Qwen/Qwen2.5-0.5B",
}
DEFAULT_MODEL = "0.5b"

# Hierarchical chunking
GLOBAL_TOKENS = 512
LOCAL_CHUNK_TOKENS = 1024
MAX_CHUNKS = 20
CHUNK_MAX_LEN = max(GLOBAL_TOKENS, LOCAL_CHUNK_TOKENS)

# Head (train-lee)
HEAD_DIM = 256
NUM_QUERIES_PER_JUSTICE = 8
SELF_ATTN_LAYERS = 2
SELF_ATTN_FFN_DIM = 1024
MAX_JUSTICES = 128

OUTPUT_DIR = "output/scotus-frozen-hierarchical"
RESULTS_DIR = "results"
EVAL_YEAR = "2019"
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
NUM_EPOCHS = 3


# ── Data loading & vote parsing ────────────────────────────────────────────────

def load_transcripts(data_dir: str) -> list[dict]:
    """Load transcript files. Returns list of {transcript, votes, filename}."""
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
        votes_text = text[idx + len(VOTES_DELIMITER) :].strip()
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
    """Parse vote lines into {justice_name: 0|1}. 0=Petitioner, 1=Respondent (train-lee)."""
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


def split_by_year(samples: list[dict], eval_year: str) -> tuple[list[dict], list[dict]]:
    train_samples = [s for s in samples if not s["filename"].startswith(eval_year)]
    eval_samples = [s for s in samples if s["filename"].startswith(eval_year)]
    print(f"Split: {len(train_samples)} train, {len(eval_samples)} eval (year={eval_year})")
    return train_samples, eval_samples


def case_result(votes: dict[str, int]) -> int | None:
    """Case winner by majority. 0=Petitioner, 1=Respondent."""
    if not votes:
        return None
    pet = sum(1 for v in votes.values() if v == 0)
    resp = sum(1 for v in votes.values() if v == 1)
    if pet > resp:
        return 0
    if resp > pet:
        return 1
    return None


# ── Justice Registry ──────────────────────────────────────────────────────────

class JusticeRegistry:
    """Maps justice names to integer IDs, discovered from data (train-lee)."""

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


# ── Hierarchical chunking ──────────────────────────────────────────────────────

def hierarchical_chunk_token_ids(
    token_ids: list[int],
    global_tokens: int,
    local_chunk_tokens: int,
    max_chunks: int,
) -> list[list[int]]:
    """Split into [global_start, middle_chunks..., global_end], capped at max_chunks."""
    n = len(token_ids)
    if n <= global_tokens:
        return [token_ids]
    start = token_ids[:global_tokens]
    end = token_ids[-global_tokens:] if n > global_tokens else []
    middle_start = global_tokens
    middle_end = n - global_tokens if n > 2 * global_tokens else global_tokens
    middle = token_ids[middle_start:middle_end] if middle_end > middle_start else []
    chunks = [start]
    for i in range(0, len(middle), local_chunk_tokens):
        chunks.append(middle[i : i + local_chunk_tokens])
    if end:
        chunks.append(end)
    if len(chunks) > max_chunks:
        chunks = chunks[:1] + chunks[1 : max_chunks - 1] + chunks[-1:]
    return chunks


# ── Dataset ────────────────────────────────────────────────────────────────────

class HierarchicalChunkDataset(Dataset):
    """Tokenize transcript, hierarchical chunk, return chunks + votes dict."""

    def __init__(
        self,
        samples: list[dict],
        tokenizer,
        global_tokens: int = GLOBAL_TOKENS,
        local_chunk_tokens: int = LOCAL_CHUNK_TOKENS,
        max_chunks: int = MAX_CHUNKS,
        chunk_max_len: int = CHUNK_MAX_LEN,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.global_tokens = global_tokens
        self.local_chunk_tokens = local_chunk_tokens
        self.max_chunks = max_chunks
        self.chunk_max_len = chunk_max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        token_ids = self.tokenizer.encode(
            s["transcript"],
            add_special_tokens=True,
            truncation=True,
            max_length=100000,
        )
        chunks = hierarchical_chunk_token_ids(
            token_ids,
            self.global_tokens,
            self.local_chunk_tokens,
            self.max_chunks,
        )
        return {"chunks": chunks, "votes": s["votes"]}


def collate_hierarchical(batch, chunk_max_len: int, max_chunks: int, pad_token_id: int):
    """Pad chunks; return chunk tensors + list of votes dicts per sample."""
    num_samples = len(batch)
    chunk_input_ids = torch.full(
        (num_samples, max_chunks, chunk_max_len),
        pad_token_id,
        dtype=torch.long,
    )
    chunk_attention_mask = torch.zeros(num_samples, max_chunks, chunk_max_len, dtype=torch.long)
    chunk_mask = torch.zeros(num_samples, max_chunks, dtype=torch.long)
    votes_list = []

    for i, b in enumerate(batch):
        votes_list.append(b["votes"])
        chunks = b["chunks"]
        n = min(len(chunks), max_chunks)
        chunk_mask[i, :n] = 1
        for c in range(n):
            ids = chunks[c]
            if len(ids) > chunk_max_len:
                ids = ids[-chunk_max_len:]
            L = len(ids)
            chunk_input_ids[i, c, :L] = torch.tensor(ids, dtype=torch.long)
            chunk_attention_mask[i, c, :L] = 1

    return {
        "chunk_input_ids": chunk_input_ids,
        "chunk_attention_mask": chunk_attention_mask,
        "chunk_mask": chunk_mask,
        "votes_list": votes_list,
    }


# ── Model components (train-lee) ───────────────────────────────────────────────

class LayerWeighter(nn.Module):
    """Learns softmax weights over LLM hidden layers (ELMo-style)."""

    def __init__(self, num_layers: int):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))

    def weights(self) -> torch.Tensor:
        return F.softmax(self.layer_logits, dim=0)


class JusticeEmbeddings(nn.Module):
    """Learnable query vectors per justice."""

    def __init__(self, max_justices: int, num_queries: int, dim: int):
        super().__init__()
        self.num_queries = num_queries
        self.embedding = nn.Embedding(max_justices * num_queries, dim)

    def forward(self, justice_ids: torch.Tensor) -> torch.Tensor:
        J = justice_ids.shape[0]
        base = justice_ids.unsqueeze(1) * self.num_queries
        offsets = torch.arange(self.num_queries, device=justice_ids.device)
        indices = base + offsets
        return self.embedding(indices)  # (J, Q, dim)


class ChunkCrossAttention(nn.Module):
    """Cross-attention from justice queries to chunk embeddings (with chunk mask)."""

    def __init__(self, llm_dim: int, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.kv_proj = nn.Linear(llm_dim, head_dim * 2, bias=False)
        self.out_proj = nn.Linear(head_dim, head_dim, bias=False)
        self.layer_norm = nn.LayerNorm(head_dim)

    def forward(
        self,
        justice_queries: torch.Tensor,
        chunk_embeddings: torch.Tensor,
        chunk_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        justice_queries: (J, Q, head_dim)
        chunk_embeddings: (1, num_chunks, llm_dim)
        chunk_mask: (num_chunks,) 1=real, 0=pad. Padded positions masked in attention.
        Returns: (J, Q, head_dim)
        """
        J, Q, D = justice_queries.shape
        S = chunk_embeddings.shape[1]
        if chunk_mask is not None:
            chunk_mask = chunk_mask.to(chunk_embeddings.device)
        kv = self.kv_proj(chunk_embeddings.squeeze(0))  # (S, D*2)
        k, v = kv.chunk(2, dim=-1)  # each (S, D)
        q = justice_queries.reshape(J * Q, 1, D)
        k = k.unsqueeze(0).expand(J * Q, -1, -1)
        v = v.unsqueeze(0).expand(J * Q, -1, -1)
        if chunk_mask is not None:
            attn_mask = torch.where(
                chunk_mask.bool(),
                torch.zeros(1, S, device=chunk_embeddings.device, dtype=k.dtype),
                torch.full((1, S), float("-inf"), device=chunk_embeddings.device, dtype=k.dtype),
            )
        else:
            attn_mask = None
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.view(J, Q, D)
        out = self.out_proj(attn_out.view(J * Q, D)).view(J, Q, D)
        out = self.layer_norm((out + justice_queries).view(J * Q, D)).view(J, Q, D)
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
        pooled = justice_repr.mean(dim=1)
        normed = self.layer_norm(pooled)
        return self.classifier(normed)


class SCOTUSVoteHead(nn.Module):
    """Train-lee style head: justice embeddings, cross-attn to chunks, self-attn, classifier."""

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
        self.cross_attention = ChunkCrossAttention(llm_dim, head_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=head_dim,
            nhead=num_queries,
            dim_feedforward=self_attn_ffn_dim,
            batch_first=True,
            norm_first=True,
        )
        self.self_attention = nn.TransformerEncoder(encoder_layer, num_layers=self_attn_layers)
        self.vote_classifier = VoteClassifier(num_queries, head_dim)

    def forward(
        self,
        chunk_embeddings: torch.Tensor,
        chunk_mask: torch.Tensor,
        justice_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        chunk_embeddings: (1, num_chunks, llm_dim)
        chunk_mask: (num_chunks,) 1=real, 0=pad
        justice_ids: (J,)
        Returns: logits (J, 2)
        """
        queries = self.justice_embeddings(justice_ids)  # (J, Q, head_dim)
        cross_out = self.cross_attention(queries, chunk_embeddings, chunk_mask)  # (J, Q, head_dim)
        J, Q, D = cross_out.shape
        flat = cross_out.view(1, J * Q, D)
        self_out = self.self_attention(flat)
        self_out = self_out.squeeze(0).view(J, Q, D)
        logits = self.vote_classifier(self_out)  # (J, 2)
        return logits

    def get_layer_weights(self) -> torch.Tensor:
        with torch.no_grad():
            return self.layer_weighter.weights()


# ── Backbone & chunk encoding ──────────────────────────────────────────────────

def load_frozen_backbone(model_name: str, quantize: bool = True):
    """Load Qwen2.5 frozen, with optional 4-bit (train-lee)."""
    print(f"Loading frozen backbone: {model_name} ({'4-bit' if quantize else 'bf16'})")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs = dict(attn_implementation="sdpa", trust_remote_code=True)
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
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model, tokenizer


def get_blended_chunk_embeddings(
    backbone,
    layer_weighter: LayerWeighter,
    chunk_input_ids: torch.Tensor,
    chunk_attention_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Run backbone on all chunks (B*max_chunks, chunk_len), blend layers, mean-pool per chunk.
    Returns chunk_embeddings: (B, max_chunks, H). Caller uses chunk_attention_mask for chunk_mask.
    """
    B, max_chunks, chunk_len = chunk_input_ids.shape
    flat_ids = chunk_input_ids.view(B * max_chunks, chunk_len)
    flat_attn = chunk_attention_mask.view(B * max_chunks, chunk_len)
    flat_ids = flat_ids.to(device)
    flat_attn = flat_attn.to(device)

    with torch.no_grad():
        outputs = backbone(
            input_ids=flat_ids,
            attention_mask=flat_attn,
            output_hidden_states=True,
        )

    weights = layer_weighter.weights()
    blended = None
    for i, h in enumerate(outputs.hidden_states):
        weighted = weights[i] * h.detach().float()
        if blended is None:
            blended = weighted
        else:
            blended = blended + weighted
    del outputs

    mask = flat_attn.unsqueeze(-1).to(blended.dtype)
    pooled = (blended * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # (B*max_chunks, H)
    chunk_embeddings = pooled.view(B, max_chunks, -1).to(device)
    return chunk_embeddings


# ── Training ───────────────────────────────────────────────────────────────────

def train_epoch(
    head: SCOTUSVoteHead,
    backbone,
    layer_weighter: LayerWeighter,
    train_loader,
    optimizer,
    scheduler,
    registry: JusticeRegistry,
    device: torch.device,
    global_step_ref: list,
    grad_accum_steps: int,
) -> dict:
    """Run one training epoch. Returns dict with train_loss, train_vote_acc, train_case_acc."""
    head.train()
    accum_loss = 0.0
    accum_correct = 0
    accum_total = 0
    accum_case_correct = 0
    accum_cases = 0
    micro_step = 0
    # Epoch-level totals for final stats
    epoch_loss_sum = 0.0
    epoch_loss_n = 0
    epoch_correct = 0
    epoch_votes = 0
    epoch_case_correct = 0
    epoch_cases = 0

    pbar = tqdm(train_loader, desc="Train")
    for batch in pbar:
        chunk_input_ids = batch["chunk_input_ids"]
        chunk_attention_mask = batch["chunk_attention_mask"]
        chunk_mask_batch = (chunk_attention_mask.sum(dim=2) > 0).long()
        votes_list = batch["votes_list"]

        chunk_embeddings = get_blended_chunk_embeddings(
            backbone, layer_weighter, chunk_input_ids, chunk_attention_mask, device
        )
        B = chunk_embeddings.shape[0]

        batch_loss_tensor = torch.tensor(0.0, device=device)
        batch_loss = 0.0
        for i in range(B):
            justice_names = list(votes_list[i].keys())
            justice_ids = torch.tensor(
                [registry.get_or_add(n) for n in justice_names],
                device=device,
                dtype=torch.long,
            )
            labels = torch.tensor(
                [votes_list[i][n] for n in justice_names],
                device=device,
                dtype=torch.long,
            )
            blended_i = chunk_embeddings[i : i + 1]
            mask_i = chunk_mask_batch[i]
            logits = head(blended_i, mask_i, justice_ids)
            loss = F.cross_entropy(logits, labels) / (grad_accum_steps * B)
            batch_loss_tensor = batch_loss_tensor + loss
            batch_loss += loss.item() * grad_accum_steps * B
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                accum_correct += (preds == labels).sum().item()
                accum_total += len(labels)
                epoch_correct += (preds == labels).sum().item()
                epoch_votes += len(labels)
                pred_maj = 0 if (preds == 0).sum() > (preds == 1).sum() else 1
                true_maj = case_result(votes_list[i])
                if true_maj is not None:
                    accum_case_correct += int(pred_maj == true_maj)
                    accum_cases += 1
                    epoch_case_correct += int(pred_maj == true_maj)
                    epoch_cases += 1
        batch_loss_tensor.backward()

        accum_loss += batch_loss / B
        epoch_loss_sum += batch_loss / B
        epoch_loss_n += 1
        micro_step += 1

        if micro_step % grad_accum_steps == 0:
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            global_step_ref[0] += 1
            pbar.set_postfix(
                loss=f"{accum_loss / micro_step:.4f}",
                acc=f"{accum_correct / max(accum_total, 1):.3f}",
                case=f"{accum_case_correct / max(accum_cases, 1):.3f}",
            )
            accum_loss = 0.0
            accum_correct = 0
            accum_total = 0
            accum_case_correct = 0
            accum_cases = 0
            micro_step = 0

    if micro_step > 0:
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        global_step_ref[0] += 1

    return {
        "train_loss": epoch_loss_sum / max(epoch_loss_n, 1),
        "train_vote_acc": epoch_correct / max(epoch_votes, 1),
        "train_case_acc": epoch_case_correct / max(epoch_cases, 1),
    }


@torch.no_grad()
def evaluate(
    head: SCOTUSVoteHead,
    backbone,
    layer_weighter: LayerWeighter,
    eval_loader,
    registry: JusticeRegistry,
    device: torch.device,
):
    head.eval()
    total_correct = 0
    total_votes = 0
    case_correct = 0
    case_count = 0

    for batch in tqdm(eval_loader, desc="Eval"):
        chunk_embeddings = get_blended_chunk_embeddings(
            backbone, layer_weighter,
            batch["chunk_input_ids"], batch["chunk_attention_mask"],
            device,
        )
        chunk_mask_batch = (batch["chunk_attention_mask"].sum(dim=2) > 0).long()
        B = chunk_embeddings.shape[0]
        for i in range(B):
            justice_names = list(batch["votes_list"][i].keys())
            justice_ids = torch.tensor(
                [registry.get_or_add(n) for n in justice_names],
                device=device,
                dtype=torch.long,
            )
            labels = torch.tensor(
                [batch["votes_list"][i][n] for n in justice_names],
                device=device,
                dtype=torch.long,
            )
            logits = head(chunk_embeddings[i : i + 1], chunk_mask_batch[i], justice_ids)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_votes += len(labels)
            pred_maj = 0 if (preds == 0).sum() > (preds == 1).sum() else 1
            true_maj = case_result(batch["votes_list"][i])
            if true_maj is not None:
                case_correct += int(pred_maj == true_maj)
                case_count += 1

    head.train()
    vote_acc = total_correct / max(total_votes, 1)
    case_acc = case_correct / max(case_count, 1)
    return vote_acc, case_acc


def save_results(
    run_config: dict,
    epochs: list[dict],
    results_dir: str,
    run_id: str | None = None,
) -> str:
    """
    Write run config and per-epoch stats to results_dir. Creates the dir if needed.
    run_config: e.g. model, lr, batch_size, data_dir, eval_year, ...
    epochs: list of {"epoch": 1, "train_loss": ..., "train_vote_acc": ..., "eval_vote_acc": ..., "eval_case_acc": ...}
    Returns the path to the written JSON file.
    """
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
    # Also write a flat CSV for easy plotting (one row per epoch)
    csv_path = os.path.join(results_dir, f"metrics_{run_id}.csv")
    csv_keys = ["epoch", "train_loss", "train_vote_acc", "train_case_acc", "eval_vote_acc", "eval_case_acc"]
    if epochs:
        with open(csv_path, "w") as f:
            f.write(",".join(csv_keys) + "\n")
            for row in epochs:
                f.write(",".join(str(row.get(k, "")) for k in csv_keys) + "\n")
    print(f"Results saved to {results_dir}: {json_path}, {csv_path}")
    return json_path


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical chunking + train-lee style head (frozen Qwen2.5)"
    )
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--model", choices=list(MODELS.keys()), default=DEFAULT_MODEL)
    parser.add_argument("--no-quant", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--global-tokens", type=int, default=GLOBAL_TOKENS)
    parser.add_argument("--local-chunk-tokens", type=int, default=LOCAL_CHUNK_TOKENS)
    parser.add_argument("--max-chunks", type=int, default=MAX_CHUNKS)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--eval-year", default=EVAL_YEAR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="Directory for run config and metrics (JSON + CSV)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = MODELS[args.model]
    chunk_max_len = max(args.global_tokens, args.local_chunk_tokens)

    all_samples = load_transcripts(args.data_dir)
    train_samples, eval_samples = split_by_year(all_samples, args.eval_year)
    if not train_samples:
        raise RuntimeError("No training samples")

    registry = JusticeRegistry()
    registry.build_from_samples(all_samples)

    backbone, tokenizer = load_frozen_backbone(model_name, quantize=not args.no_quant)
    num_llm_layers = backbone.config.num_hidden_layers + 1
    llm_dim = backbone.config.hidden_size
    print(f"Backbone: {num_llm_layers} layers, dim={llm_dim}")

    head = SCOTUSVoteHead(
        num_llm_layers=num_llm_layers,
        llm_dim=llm_dim,
    ).to(device).float()

    total_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"Trainable head parameters: {total_params:,}")

    train_ds = HierarchicalChunkDataset(
        train_samples,
        tokenizer,
        global_tokens=args.global_tokens,
        local_chunk_tokens=args.local_chunk_tokens,
        max_chunks=args.max_chunks,
        chunk_max_len=chunk_max_len,
    )
    eval_ds = HierarchicalChunkDataset(
        eval_samples,
        tokenizer,
        global_tokens=args.global_tokens,
        local_chunk_tokens=args.local_chunk_tokens,
        max_chunks=args.max_chunks,
        chunk_max_len=chunk_max_len,
    )

    def collate_fn(batch):
        return collate_hierarchical(
            batch, chunk_max_len, args.max_chunks, tokenizer.pad_token_id
        )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    effective_batch = args.batch_size * GRAD_ACCUM_STEPS
    total_steps = (len(train_samples) * args.epochs) // effective_batch
    warmup_steps = int(total_steps * WARMUP_RATIO)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(1e-6 / args.lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    global_step_ref = [0]

    run_config = {
        "model": args.model,
        "model_path": MODELS[args.model],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "eval_year": args.eval_year,
        "data_dir": args.data_dir,
        "global_tokens": args.global_tokens,
        "local_chunk_tokens": args.local_chunk_tokens,
        "max_chunks": args.max_chunks,
        "n_train": len(train_samples),
        "n_eval": len(eval_samples),
        "quantize": not args.no_quant,
    }
    epochs_stats = []

    for epoch in range(args.epochs):
        train_stats = train_epoch(
            head, backbone, head.layer_weighter,
            train_loader, optimizer, scheduler,
            registry, device, global_step_ref, GRAD_ACCUM_STEPS,
        )
        row = {"epoch": epoch + 1, **train_stats}
        if eval_samples:
            vote_acc, case_acc = evaluate(
                head, backbone, head.layer_weighter, eval_loader, registry, device
            )
            row["eval_vote_acc"] = vote_acc
            row["eval_case_acc"] = case_acc
            print(f"Epoch {epoch+1}  eval vote_acc={vote_acc:.4f}  case_acc={case_acc:.4f}")
        epochs_stats.append(row)

    save_results(run_config, epochs_stats, args.results_dir, run_id=run_id)

    save_path = os.path.join(args.output_dir, "vote_head_hierarchical_lee.pt")
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


if __name__ == "__main__":
    main()
