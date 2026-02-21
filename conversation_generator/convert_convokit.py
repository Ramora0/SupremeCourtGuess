"""Convert ConvoKit Supreme Court corpus into clean transcript files for training.

Each output file contains:
  <transcript>
  Justice last names for justices, side labels for advocates.
  </transcript>
  <votes>
  LastName: Petitioner/Respondent
"""

import os
import re

from convokit import Corpus, download
from tqdm import tqdm

OUTPUT_DIR = "data/transcripts"

# Side 1 = petitioner (argues first in SCOTUS), side 0 = respondent.
# Side 3 = unknown/inferred, side 2 = amicus.
PETITIONER_SIDE = 1
RESPONDENT_SIDE = 0


def clean_last_name(justice_id: str) -> str:
    """j__john_m_harlan2 -> Harlan, j__earl_warren -> Warren."""
    name = justice_id.removeprefix("j__")
    name = re.sub(r"\d+$", "", name)  # drop trailing numbers (harlan2)
    last = name.split("_")[-1]
    return last.capitalize()


def advocate_label(side: int) -> str:
    if side == PETITIONER_SIDE:
        return "Petitioner's Counsel"
    elif side == RESPONDENT_SIDE:
        return "Respondent's Counsel"
    else:
        return "Amicus Counsel"


def speaker_label(speaker_id: str, speaker_type: str | None,
                  advocates: dict) -> str | None:
    """Return the display label for a speaker, or None to skip."""
    if speaker_id == "<INAUDIBLE>":
        return None
    if speaker_type == "J":
        return clean_last_name(speaker_id)
    if speaker_id in advocates:
        return advocate_label(advocates[speaker_id].get("side", 3))
    # Unknown non-justice, non-advocate (rare)
    return None


def format_case(convo) -> str | None:
    """Build a formatted transcript string for one case, or None to skip."""
    meta = dict(convo.meta)
    votes_side = meta.get("votes_side") or {}
    win_side = meta.get("win_side")
    advocates = meta.get("advocates") or {}

    if not votes_side or win_side is None:
        return None

    # --- transcript ---
    lines = []
    for utt_id in sorted(convo.get_utterance_ids()):
        utt = convo.get_utterance(utt_id)
        text = utt.text.strip().replace("\n", " ")
        text = re.sub(r"  +", " ", text)  # collapse multiple spaces
        if not text:
            continue

        label = speaker_label(
            utt.speaker.id, utt.meta.get("speaker_type"), advocates
        )
        if label is None:
            continue

        lines.append(f"{label}: {text}")

    if not lines:
        return None

    # --- votes ---
    vote_lines = []
    for justice_id, side in sorted(votes_side.items()):
        if side == -1:  # recusal / non-participation
            continue
        last_name = clean_last_name(justice_id)
        if side == PETITIONER_SIDE:
            vote_lines.append(f"{last_name}: Petitioner")
        elif side == RESPONDENT_SIDE:
            vote_lines.append(f"{last_name}: Respondent")
        # skip any other value

    if not vote_lines:
        return None

    transcript = "\n".join(lines)
    votes = "\n".join(vote_lines)
    return f"<transcript>\n{transcript}\n</transcript>\n<votes>\n{votes}"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert ConvoKit SCOTUS corpus to transcripts")
    parser.add_argument("-n", "--num", type=int, default=None,
                        help="Number of cases to convert (default: all)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading ConvoKit Supreme Court corpus...")
    corpus = Corpus(filename=download("supreme-corpus"))
    convo_ids = corpus.get_conversation_ids()

    if args.num is not None:
        convo_ids = convo_ids[:args.num]

    print(f"Processing {len(convo_ids)} conversations...")

    written = 0
    skipped = 0

    for cid in tqdm(convo_ids, desc="Converting"):
        convo = corpus.get_conversation(cid)
        case_id = dict(convo.meta).get("case_id", str(cid))
        safe_id = case_id.replace("/", "_")

        result = format_case(convo)
        if result is None:
            skipped += 1
            continue

        path = os.path.join(OUTPUT_DIR, f"{safe_id}.txt")
        with open(path, "w") as f:
            f.write(result)
        written += 1

    print(f"\nDone. Wrote {written} files, skipped {skipped}.")
    print(f"Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
