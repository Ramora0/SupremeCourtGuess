#!/usr/bin/env python3
"""
Print a single Supreme Court case as a simple conversation:
  current_speaker
  what-they-said

(blank line between turns)

Usage:
  python case_to_conversation.py [case_id]
  MAX_UTTERANCES=50000 python case_to_conversation.py   # optional: limit memory
If case_id is omitted, uses the first case in the corpus.
"""

import os
import sys
from convokit import Corpus, download

MAX_UTTERANCES = int(os.environ.get("MAX_UTTERANCES", 0)) or None


def main():
    path = download("supreme-corpus")
    if MAX_UTTERANCES:
        corpus = Corpus(
            filename=path,
            utterance_start_index=0,
            utterance_end_index=MAX_UTTERANCES - 1,
        )
    else:
        corpus = Corpus(filename=path)

    convo_ids = corpus.get_conversation_ids()
    if not convo_ids:
        print("No conversations in corpus.", file=sys.stderr)
        sys.exit(1)

    case_id = sys.argv[1] if len(sys.argv) > 1 else convo_ids[0]
    if case_id not in convo_ids:
        print(f"Case id '{case_id}' not in corpus.", file=sys.stderr)
        sys.exit(1)

    convo = corpus.get_conversation(case_id)
    try:
        utts = convo.get_chronological_utterance_list()
    except (TypeError, ValueError):
        utts = list(convo.iter_utterances())

    for utt in utts:
        if utt.speaker:
            speaker = utt.speaker.meta.get("name") or utt.speaker.id
        else:
            speaker = "(unknown)"
        text = (utt.text or "").strip()
        print(f"{speaker}\n{text}\n")


if __name__ == "__main__":
    main()
