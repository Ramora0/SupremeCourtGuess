#!/usr/bin/env python3
"""
Export every Supreme Court case as a simple conversation to its own .txt file.
Same format as case_to_conversation.py: speaker name, then what they said, blank line between turns.
Files are saved as {case_id}.txt in an output directory (default: case_transcripts).

Usage:
  python all_cases_to_conversations.py [output_dir]
  MAX_UTTERANCES=50000 python all_cases_to_conversations.py   # optional: limit memory
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


def _speaker_display_name(convo, speaker_id):
    """Get display name for a speaker (justice or advocate) by id."""
    try:
        s = convo.get_speaker(speaker_id)
        return (s.meta.get("name") or s.id).strip() or speaker_id
    except Exception:
        return str(speaker_id)


def _build_speaker_key(convo):
    """Build header listing which speakers are on petitioner vs respondent side."""
    meta = dict(convo.meta)
    votes_side = meta.get("votes_side") or {}
    advocates = meta.get("advocates") or {}

    petitioner_names = set()
    for jid, side in votes_side.items():
        if side == PETITIONER_SIDE:
            petitioner_names.add(_speaker_display_name(convo, jid))
    for aid, info in advocates.items():
        side = info.get("side") if isinstance(info, dict) else None
        if side == PETITIONER_SIDE:
            petitioner_names.add(_speaker_display_name(convo, aid))

    respondent_names = set()
    for jid, side in votes_side.items():
        if side == RESPONDENT_SIDE:
            respondent_names.add(_speaker_display_name(convo, jid))
    for aid, info in advocates.items():
        side = info.get("side") if isinstance(info, dict) else None
        if side == RESPONDENT_SIDE:
            respondent_names.add(_speaker_display_name(convo, aid))

    lines = [
        "PETITIONER'S SIDE: " + ", ".join(sorted(petitioner_names)) if petitioner_names else "PETITIONER'S SIDE: (none listed)",
        "RESPONDENT'S SIDE: " + ", ".join(sorted(respondent_names)) if respondent_names else "RESPONDENT'S SIDE: (none listed)",
        "",
    ]
    return "\n".join(lines)


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
    """Return the conversational text for one case (same format as case_to_conversation.py), with key header and outcome footer."""
    key_header = _build_speaker_key(convo)
    try:
        utts = convo.get_chronological_utterance_list()
    except (TypeError, ValueError):
        utts = list(convo.iter_utterances())
    lines = []
    for utt in utts:
        if utt.speaker:
            speaker = utt.speaker.meta.get("name") or utt.speaker.id
        else:
            speaker = "(unknown)"
        text = (utt.text or "").strip()
        lines.append(f"{speaker}\n{text}\n")
    transcript = "\n".join(lines)
    justice_votes = _build_justice_votes(convo)
    outcome = _build_winner_footer(convo)
    return f"{key_header}{transcript}\n\n---\n{justice_votes}\n\n{outcome}"


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
        # Use case_id from metadata (e.g. docket number) for filename; fall back to conversation id
        case_id = dict(convo.meta).get("case_id", convo_id)
        name = safe_filename(case_id)
        # If multiple conversations share the same case_id (e.g. multiple sessions), disambiguate
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
