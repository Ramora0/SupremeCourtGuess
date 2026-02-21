#!/usr/bin/env python3
"""
Export every Supreme Court case as a compact conversation (variant 3):
- No key. Body uses literal labels: Respondent, Petitioner, or judge last name per turn.
- Judges: last name only.

Usage:
  python all_cases_to_convo_3.py [output_dir]
  MAX_UTTERANCES=50000 python all_cases_to_convo_3.py   # optional: limit memory
"""

import os
import random
import re
import sys
from convokit import Corpus, download
from tqdm import tqdm

MAX_UTTERANCES = int(os.environ.get("MAX_UTTERANCES", 0)) or None


def safe_filename(case_id):
    """Make case_id safe for use as a filename (e.g. slashes -> underscores)."""
    s = str(case_id)
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    return s or "unknown"


# ConvoKit side codes: 1 = petitioner, 0 = respondent, -1 = recusal/unclear
PETITIONER_SIDE = 1
RESPONDENT_SIDE = 0

def _speaker_display_name(convo, speaker_id):
    """Get display name for a speaker (justice or advocate) by id."""
    try:
        s = convo.get_speaker(speaker_id)
        return (s.meta.get("name") or s.id).strip() or speaker_id
    except Exception:
        return str(speaker_id)


def _speaker_last_name(convo, speaker_id):
    """Get last name only (for judges). Splits on space and takes last token."""
    full = _speaker_display_name(convo, speaker_id)
    parts = full.split()
    return parts[-1] if parts else full


def _build_speaker_to_label(convo):
    """
    Build speaker_id -> label mapping. Labels: Respondent, Petitioner, or judge last name.
    Returns dict speaker_id -> str.
    """
    votes_side = convo.meta.get("votes_side") or {}
    advocates = convo.meta.get("advocates") or {}
    speaker_to_label = {}

    # Judges: last name
    for jid in sorted(votes_side.keys()):
        speaker_to_label[jid] = _speaker_last_name(convo, jid)

    # Advocates: Respondent or Petitioner by side
    for aid, info in (advocates or {}).items():
        if aid in speaker_to_label:
            continue
        side = info.get("side") if isinstance(info, dict) else None
        if side == PETITIONER_SIDE:
            speaker_to_label[aid] = "Petitioner"
        elif side == RESPONDENT_SIDE:
            speaker_to_label[aid] = "Respondent"
        else:
            speaker_to_label[aid] = "Petitioner"

    return speaker_to_label


def _speaker_to_label(convo, utt, speaker_to_label):
    """Resolve utterance speaker to label (Respondent, Petitioner, or judge last name)."""
    if utt.speaker is None:
        return "Petitioner"
    sid = utt.speaker.id
    if sid in speaker_to_label:
        return speaker_to_label[sid]
    if sid in (convo.meta.get("votes_side") or {}):
        return _speaker_last_name(convo, sid)
    return "Petitioner"


def _build_justice_votes(convo):
    """Build list of each justice and how they voted. Uses last name for justices."""
    votes_side = convo.meta.get("votes_side") or {}
    if not votes_side:
        return "JUSTICE VOTES: (none in metadata)"
    lines = ["JUSTICE VOTES:"]
    jids = sorted(votes_side.keys())
    random.shuffle(jids)
    for jid in jids:
        side = votes_side[jid]
        name = _speaker_last_name(convo, jid)
        if side == PETITIONER_SIDE:
            lines.append(f"{name}: Petitioner")
        elif side == RESPONDENT_SIDE:
            lines.append(f"{name}: Respondent")
        else:
            lines.append(f"{name}: Unknown")
    return "\n".join(lines)


def _build_winner_footer(convo):
    """Build footer stating who won the case. win_side: 1=petitioner, 0=respondent, 2=unclear, -1=unavailable."""
    win_side = convo.meta.get("win_side")
    if win_side == 1:
        return "OUTCOME: Petitioner won."
    if win_side == 0:
        return "OUTCOME: Respondent won."
    if win_side == 2:
        return "OUTCOME: Unclear."
    return "OUTCOME: Unknown."


def case_to_text(convo):
    """Return transcript: body only (speaker label + text per turn), then footer. No key."""
    speaker_to_label = _build_speaker_to_label(convo)
    try:
        utts = convo.get_chronological_utterance_list()
    except (TypeError, ValueError):
        utts = list(convo.iter_utterances())

    turn_lines = []
    for utt in utts:
        label = _speaker_to_label(convo, utt, speaker_to_label)
        text = (utt.text or "").strip()
        turn_lines.append(f"{label}\n{text}")
    transcript = "\n".join(turn_lines)
    justice_votes = _build_justice_votes(convo)
    outcome = _build_winner_footer(convo)
    return f"{transcript}\n\n---\n{justice_votes}\n\n{outcome}"


def main():
    path = download("supreme-corpus")
    if MAX_UTTERANCES:
        print(f"Loading corpus (first {MAX_UTTERANCES} utterances)...", file=sys.stderr)
        corpus = Corpus(
            filename=path,
            utterance_start_index=0,
            utterance_end_index=MAX_UTTERANCES - 1,
        )
    else:
        print("Loading full corpus...", file=sys.stderr)
        corpus = Corpus(filename=path)

    convo_ids = corpus.get_conversation_ids()
    if not convo_ids:
        print("No conversations in corpus.", file=sys.stderr)
        sys.exit(1)

    out_dir = sys.argv[1] if len(sys.argv) > 1 else "case_transcripts"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Writing {len(convo_ids)} cases to {out_dir}/", file=sys.stderr)

    seen_case_ids = set()
    for i, convo_id in tqdm(enumerate(convo_ids, 1), total=len(convo_ids), desc="Processing cases"):
        convo = corpus.get_conversation(convo_id)
        case_id = dict(convo.meta).get("case_id", convo_id)
        name = safe_filename(case_id)
        if name in seen_case_ids:
            name = f"{name}_{safe_filename(convo_id)}"
        seen_case_ids.add(name)
        out_path = os.path.join(out_dir, f"{name}.txt")
        try:
            text = case_to_text(convo)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            print(f"  Skip {case_id}: {e}", file=sys.stderr)
            continue
        if i % 100 == 0 or i == len(convo_ids):
            print(f"  {i}/{len(convo_ids)}", file=sys.stderr)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
