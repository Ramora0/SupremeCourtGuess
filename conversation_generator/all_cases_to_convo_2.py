#!/usr/bin/env python3
"""
Export every Supreme Court case as a compact conversation: speaker key (Petitioner, Respondent, judges by name)
then transcript body using only indices. Minimizes string length without losing data.

Key format: 0=Respondent|1=Petitioner|2=JudgeName|... (indices align with ConvoKit side codes)
Body: each turn is "index\\ntext\\n" (no advocate names, judge identity via key).

Usage:
  python all_cases_to_convo_2.py [output_dir]
  MAX_UTTERANCES=50000 python all_cases_to_convo_2.py   # optional: limit memory
"""

import os
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

# Compact key indices align with ConvoKit side codes: 0 = respondent, 1 = petitioner, 2+ = judges
PETITIONER_INDEX = 1
RESPONDENT_INDEX = 0
JUDGE_START_INDEX = 2


def _speaker_display_name(convo, speaker_id):
    """Get display name for a speaker (justice or advocate) by id."""
    try:
        s = convo.get_speaker(speaker_id)
        return (s.meta.get("name") or s.id).strip() or speaker_id
    except Exception:
        return str(speaker_id)


def _build_compact_key_and_mapping(convo):
    """
    Build compact key string and speaker_id -> index mapping.
    Key: 0=Respondent|1=Petitioner|2=JudgeName1|3=JudgeName2|...
    Mapping: advocate side (ConvoKit 0/1) -> same index; justice -> 2, 3, ...
    Returns (key_line, speaker_id_to_index).
    """
    votes_side = convo.meta.get("votes_side") or {}
    advocates = convo.meta.get("advocates") or {}

    # Judges in stable order (sorted by id), each gets an index starting at JUDGE_START_INDEX
    judge_ids = sorted(votes_side.keys())
    key_parts = ["0=Respondent", "1=Petitioner"]
    speaker_id_to_index = {}

    for i, jid in enumerate(judge_ids):
        idx = JUDGE_START_INDEX + i
        name = _speaker_display_name(convo, jid)
        key_parts.append(f"{idx}={name}")
        speaker_id_to_index[jid] = idx

    # Advocates: map by side only (no identity); skip if already a judge
    for aid, info in (advocates or {}).items():
        if aid in speaker_id_to_index:
            continue  # already mapped as judge
        side = info.get("side") if isinstance(info, dict) else None
        if side == PETITIONER_SIDE:
            speaker_id_to_index[aid] = PETITIONER_INDEX
        elif side == RESPONDENT_SIDE:
            speaker_id_to_index[aid] = RESPONDENT_INDEX
        else:
            speaker_id_to_index[aid] = PETITIONER_INDEX  # fallback

    key_line = "|".join(key_parts)
    return key_line, speaker_id_to_index


def _speaker_to_index(convo, utt, speaker_id_to_index):
    """Resolve utterance speaker to compact key index (0=Respondent, 1=Petitioner, 2+=judge)."""
    if utt.speaker is None:
        return PETITIONER_INDEX
    sid = utt.speaker.id
    if sid in speaker_id_to_index:
        return speaker_id_to_index[sid]
    # Unknown speaker: use fallback
    votes_side = convo.meta.get("votes_side") or {}
    if sid in votes_side:
        return PETITIONER_INDEX
    return PETITIONER_INDEX


def _build_justice_votes(convo):
    """Build list of each justice and how they voted (Petitioner / Respondent / Recused)."""
    votes_side = convo.meta.get("votes_side") or {}
    if not votes_side:
        return "JUSTICE VOTES: (none in metadata)"
    lines = ["JUSTICE VOTES:"]
    for jid in sorted(votes_side.keys()):
        side = votes_side[jid]
        name = _speaker_display_name(convo, jid)
        if side == PETITIONER_SIDE:
            lines.append(f"{name}: Petitioner")
        elif side == RESPONDENT_SIDE:
            lines.append(f"{name}: Respondent")
        else:
            lines.append(f"{name}: Recused / Unknown")
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
    """Return compact conversational text: key line, then index\\ntext\\n per turn, then footer."""
    key_line, speaker_id_to_index = _build_compact_key_and_mapping(convo)
    try:
        utts = convo.get_chronological_utterance_list()
    except (TypeError, ValueError):
        utts = list(convo.iter_utterances())

    turn_lines = []
    for utt in utts:
        idx = _speaker_to_index(convo, utt, speaker_id_to_index)
        text = (utt.text or "").strip()
        turn_lines.append(f"{idx}\n{text}")
    transcript = "\n".join(turn_lines)
    justice_votes = _build_justice_votes(convo)
    outcome = _build_winner_footer(convo)
    return f"{key_line}\n\n{transcript}\n\n---\n{justice_votes}\n\n{outcome}"


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
