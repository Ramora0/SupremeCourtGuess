"""Load a trained encoder checkpoint and predict votes on transcript files.

Usage:
    python predict.py output/encoder-modernbert-base.pt case_transcripts_cleaned/2019_*.txt
    python predict.py output/my-run.pt new_case.txt --justices Kagan,Gorsuch,Thomas
"""

import argparse
import importlib
import sys
from contextlib import nullcontext

import torch
from transformers import AutoModel, AutoTokenizer

# train-encoder.py has a hyphen — import via importlib
_te = importlib.import_module("train-encoder")
ENCODER_MODELS = _te.ENCODER_MODELS
MAX_JUSTICES = _te.MAX_JUSTICES
KNOWN_JUSTICE_NAMES = _te.KNOWN_JUSTICE_NAMES
SPEAKER_TYPE_MAP = _te.SPEAKER_TYPE_MAP
VOTES_DELIMITER = _te.VOTES_DELIMITER
Turn = _te.Turn
JusticeRegistry = _te.JusticeRegistry
SCOTUSEncoderModel = _te.SCOTUSEncoderModel
parse_transcript_turns = _te.parse_transcript_turns
split_into_phases = _te.split_into_phases
batch_encode_phases = _te.batch_encode_phases

# Current court (default if --justices not provided)
CURRENT_COURT = [
    "Roberts", "Thomas", "Alito", "Sotomayor", "Kagan",
    "Gorsuch", "Kavanaugh", "Barrett", "Jackson",
]


def load_checkpoint(ckpt_path: str, device: torch.device):
    """Load checkpoint and reconstruct encoder + head."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    # Load encoder
    encoder_name = cfg["encoder"]
    model_path = ENCODER_MODELS[encoder_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoder = AutoModel.from_pretrained(model_path)
    if "encoder_state_dict" in ckpt:
        encoder.load_state_dict(ckpt["encoder_state_dict"])
    else:
        print("Warning: checkpoint has no encoder_state_dict, using pretrained weights")
    encoder.to(device).eval()

    max_length = getattr(encoder.config, "max_position_embeddings", 512)

    # Build head
    model = SCOTUSEncoderModel(
        encoder_dim=cfg["encoder_dim"],
        head_dim=cfg["head_dim"],
        num_queries=cfg["num_queries"],
        self_attn_layers=cfg["self_attn_layers"],
        ffn_dim=cfg["ffn_dim"],
        dropout=cfg["dropout"],
        max_justices=cfg.get("max_justices", MAX_JUSTICES),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    # Restore registry
    registry = JusticeRegistry()
    for name, id_ in ckpt["registry"].items():
        registry.name_to_id[name] = id_
        registry.id_to_name[id_] = name
        registry._next_id = max(registry._next_id, id_ + 1)

    return encoder, tokenizer, model, registry, max_length


def load_transcript_file(path: str) -> str:
    """Read a transcript file, stripping votes section if present."""
    with open(path) as f:
        text = f.read()
    if VOTES_DELIMITER in text:
        text = text[:text.index(VOTES_DELIMITER)]
    return text


@torch.no_grad()
def predict_case(
    transcript: str,
    justice_names: list[str],
    encoder,
    tokenizer,
    model: SCOTUSEncoderModel,
    registry: JusticeRegistry,
    max_length: int,
    device: torch.device,
    amp_ctx,
) -> list[tuple[str, str, float]]:
    """Predict votes for given justices on a transcript.

    Returns list of (justice_name, predicted_vote, confidence).
    """
    turns = parse_transcript_turns(transcript)
    pet_turns, resp_turns = split_into_phases(turns)

    all_turns = pet_turns + resp_turns
    if not all_turns:
        print("  Warning: no turns parsed from transcript")
        return [(name, "Unknown", 0.5) for name in justice_names]

    # Encode phases
    phase_lists = []
    if pet_turns:
        phase_lists.append(pet_turns)
    if resp_turns:
        phase_lists.append(resp_turns)

    phase_vectors = batch_encode_phases(
        encoder, tokenizer, phase_lists, max_length, device, amp_ctx,
    )

    # Concatenate turn vectors
    turn_vectors = torch.cat(phase_vectors, dim=0)  # (T, hidden_dim)
    speaker_types = [SPEAKER_TYPE_MAP.get(t.speaker_type, 0) for t in all_turns]
    turn_speaker_types = torch.tensor(speaker_types, device=device)

    # Build per-turn justice IDs
    justice_ids_per_turn = []
    for t in all_turns:
        if t.speaker_type == "justice":
            justice_ids_per_turn.append(registry.get_or_add(t.speaker_label))
        else:
            justice_ids_per_turn.append(-1)
    turn_justice_ids = torch.tensor(justice_ids_per_turn, device=device, dtype=torch.long)

    # Justice IDs for the justices we want to predict
    j_ids = torch.tensor(
        [registry.get_or_add(n) for n in justice_names], device=device
    )

    logits, _ = model(
        turn_vectors, turn_speaker_types, j_ids,
        turn_justice_ids=turn_justice_ids,
    )

    probs = logits.softmax(dim=1)  # (J, 2)
    preds = logits.argmax(dim=1)

    results = []
    for j, name in enumerate(justice_names):
        vote = "Petitioner" if preds[j].item() == 0 else "Respondent"
        conf = probs[j, preds[j]].item()
        results.append((name, vote, conf))

    return results


def main():
    p = argparse.ArgumentParser(description="Predict SCOTUS votes from checkpoint")
    p.add_argument("checkpoint", help="Path to .pt checkpoint file")
    p.add_argument("files", nargs="+", help="Transcript file(s) to predict on")
    p.add_argument(
        "--justices", type=str, default=None,
        help="Comma-separated justice names (default: current court)",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    print(f"Loading checkpoint: {args.checkpoint}")
    encoder, tokenizer, model, registry, max_length = load_checkpoint(
        args.checkpoint, device
    )

    justice_names = (
        args.justices.split(",") if args.justices else CURRENT_COURT
    )
    print(f"Predicting for: {', '.join(justice_names)}\n")

    for path in args.files:
        print(f"── {path} ──")
        transcript = load_transcript_file(path)

        turns = parse_transcript_turns(transcript)
        pet, resp = split_into_phases(turns)
        print(f"  {len(turns)} turns ({len(pet)} pet phase, {len(resp)} resp phase)")

        results = predict_case(
            transcript, justice_names,
            encoder, tokenizer, model, registry, max_length, device, amp_ctx,
        )

        # Tally
        pet_votes = sum(1 for _, v, _ in results if v == "Petitioner")
        resp_votes = sum(1 for _, v, _ in results if v == "Respondent")
        outcome = "Petitioner" if pet_votes > resp_votes else "Respondent"

        print(f"  {'Justice':<15} {'Vote':<12} {'Confidence':>10}")
        print(f"  {'─' * 37}")
        for name, vote, conf in results:
            print(f"  {name:<15} {vote:<12} {conf:>9.1%}")
        print(f"\n  Predicted outcome: {outcome} wins ({pet_votes}-{resp_votes})\n")


if __name__ == "__main__":
    main()
