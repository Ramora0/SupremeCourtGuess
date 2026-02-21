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


def case_to_text(corpus, case_id):
    """Return the conversational text for one case (same format as case_to_conversation.py)."""
    convo = corpus.get_conversation(case_id)
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
    return "\n".join(lines)


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

    for i, case_id in tqdm(enumerate(convo_ids, 1), total=len(convo_ids), desc="Processing cases"):
        name = safe_filename(case_id)
        out_path = os.path.join(out_dir, f"{name}.txt")
        try:
            text = case_to_text(corpus, case_id)
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
