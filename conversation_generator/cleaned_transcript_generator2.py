#!/usr/bin/env python3
"""
Export every Supreme Court case as a cleaned conversation transcript (variant 2).

Same as cleaned_transcript_generator.py except:
- Prepends decided date at the beginning (from ConvoKit cases.jsonl).
- Strips name suffixes (Jr., II, III, IV, Sr.) from justice last names.

Usage:
  python cleaned_transcript_generator2.py [output_dir]
  MAX_UTTERANCES=50000 python cleaned_transcript_generator2.py [output_dir]
  python cleaned_transcript_generator2.py --merge-same-speaker --no-normalize-speakers out/
"""

import os
import random
import re
import sys
from pathlib import Path

from convokit import Corpus, download
from tqdm import tqdm

from corpus_case_info import download_cases_jsonl, get_case_legal_info_from_corpus
from scdb_matcher import format_scdb_header, get_scdb_case_row

MAX_UTTERANCES = int(os.environ.get("MAX_UTTERANCES", 0)) or None

# ConvoKit side codes
PETITIONER_SIDE = 1
RESPONDENT_SIDE = 0

# Speaker labels and known justice last names (for parsing transcript lines)
PETITIONER = "Petitioner"
RESPONDENT = "Respondent"
KNOWN_JUSTICE_LAST_NAMES = {
    "Alito", "Baldwin", "Barbour", "Barrett", "Black", "Blackmun", "Blair", "Blatchford",
    "Bradley", "Brandeis", "Brennan", "Brewer", "Breyer", "Brown", "Burger", "Burton",
    "Butler", "Campbell", "Cardozo", "Catron", "Chase", "Clark", "Clarke", "Clifford",
    "Curtis", "Daniel", "Davis", "Day", "Douglas", "Duvall", "Ellsworth", "Field",
    "Fortas", "Frankfurter", "Fuller", "Ginsburg", "Goldberg", "Gorsuch", "Gray",
    "Grier", "Harlan", "Holmes", "Hughes", "Hunt", "Iredell", "Jackson", "Jay",
    "Johnson", "Kagan", "Kavanaugh", "Kennedy", "Lamar", "Livingston", "Lurton",
    "Marshall", "Matthews", "McKenna", "McKinley", "McLean", "McReynolds",
    "Miller", "Minton", "Moody", "Moore", "Murphy", "Nelson", "O'Connor", "Paterson",
    "Peckham", "Pitney", "Powell", "Reed", "Rehnquist", "Roberts", "Rutledge",
    "Sanford", "Scalia", "Shiras", "Sotomayor", "Souter", "Stevens", "Stewart",
    "Stone", "Story", "Strong", "Sutherland", "Swayne", "Taft", "Taney", "Thomas",
    "Thompson", "Todd", "Trimble", "VanDevanter", "Vinson", "Waite", "Warren",
    "Washington", "Wayne", "White", "Whittaker", "Wilson", "Woodbury", "Woods",
}
NAME_SUFFIXES = frozenset({"Jr.", "Jr", "Sr.", "Sr", "II", "III", "IV", "V"})
# ConvoKit uses "Unknown" for unrecognized speakers; must be recognized when re-parsing
UNKNOWN_SPEAKER_LABELS = {"Unknown"}
KNOWN_SPEAKERS = {PETITIONER, RESPONDENT, "Jr."} | KNOWN_JUSTICE_LAST_NAMES | UNKNOWN_SPEAKER_LABELS


# ---------------------------------------------------------------------------
# Convo -> raw transcript (from all_cases_to_convo_3)
# ---------------------------------------------------------------------------

def safe_filename(case_id: str) -> str:
    """Make case_id safe for filenames (e.g. slashes -> underscores)."""
    s = str(case_id)
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    return s or "unknown"


def _strip_name_suffixes(name: str) -> str:
    """Remove trailing name suffix tokens (Jr., II, III, IV, Sr., etc.)."""
    if not name or not name.strip():
        return name
    parts = name.strip().split()
    while parts:
        last = parts[-1].rstrip(".,")
        if last in NAME_SUFFIXES:
            parts.pop()
        else:
            break
    return " ".join(parts).strip() or name.strip()


def _speaker_display_name(convo, speaker_id) -> str:
    """Return display name for a speaker (justice or advocate) by id."""
    try:
        s = convo.get_speaker(speaker_id)
        return (s.meta.get("name") or s.id).strip() or speaker_id
    except Exception:
        return str(speaker_id)


def _speaker_last_name(convo, speaker_id) -> str:
    """Return last name only for a speaker (splits on space, takes last token); strips Jr., II, III, etc."""
    full = _speaker_display_name(convo, speaker_id)
    full = _strip_name_suffixes(full)
    if "," in full:
        segment = full.split(",")[0].strip()
        parts = segment.split()
        return parts[-1] if parts else segment
    parts = full.split()
    return parts[-1] if parts else full


def _build_speaker_to_label(convo) -> dict:
    """Build speaker_id -> label map: Respondent, Petitioner, or judge last name."""
    votes_side = convo.meta.get("votes_side") or {}
    advocates = convo.meta.get("advocates") or {}
    speaker_to_label = {}

    for jid in sorted(votes_side.keys()):
        speaker_to_label[jid] = _speaker_last_name(convo, jid)

    for aid, info in (advocates or {}).items():
        if aid in speaker_to_label:
            continue
        side = info.get("side") if isinstance(info, dict) else None
        if side == PETITIONER_SIDE:
            speaker_to_label[aid] = "Petitioner"
        elif side == RESPONDENT_SIDE:
            speaker_to_label[aid] = "Respondent"
        else:
            speaker_to_label[aid] = "Unknown"

    return speaker_to_label


def _speaker_to_label(convo, utt, speaker_to_label: dict) -> str:
    """Resolve utterance speaker to label (Respondent, Petitioner, or judge last name)."""
    if utt.speaker is None:
        return "Unknown"
    sid = utt.speaker.id
    if sid in speaker_to_label:
        return speaker_to_label[sid]
    if sid in (convo.meta.get("votes_side") or {}):
        return _speaker_last_name(convo, sid)
    return "Unknown"


def _build_justice_votes(convo) -> str:
    """Build 'JUSTICE VOTES:' block with each justice's vote (last name only)."""
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


def _build_winner_footer(convo) -> str:
    """Build 'OUTCOME: ...' line (win_side: 1=petitioner, 0=respondent, 2=unclear)."""
    win_side = convo.meta.get("win_side")
    if win_side == 1:
        return "OUTCOME: Petitioner won."
    if win_side == 0:
        return "OUTCOME: Respondent won."
    if win_side == 2:
        return "OUTCOME: Unclear."
    return "OUTCOME: Unknown."


def convo_to_raw_transcript(convo) -> str:
    """Convert one ConvoKit conversation to raw transcript (Speaker\\nText per turn + footer)."""
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


# ---------------------------------------------------------------------------
# Transcript cleaning (from transcript_cleaner_1)
# ---------------------------------------------------------------------------

def _sub_all(pattern: str, sub: str, text: str) -> str:
    """Apply re.sub repeatedly until text stops changing (for stutter patterns)."""
    new_text = re.sub(pattern, sub, text, flags=re.IGNORECASE)
    while new_text != text:
        text = new_text
        new_text = re.sub(pattern, sub, text, flags=re.IGNORECASE)
    return text


def format_text(text: str) -> str:
    """Strip, collapse newlines/spaces, remove 'word -- word' stutters."""
    if not text or not text.strip():
        return text.strip()
    text = text.strip()
    text = _sub_all(r"\n", " ", text)
    text = _sub_all(r"  +", " ", text)
    text = _sub_all(r"\b(\w+)\s--\s\1\b", r"\1", text)
    text = _sub_all(r"\b(\w+)\s(\w+)\s--\s\1\s\2\b", r"\1 \2", text)
    return text.strip()


def format_speaker(speaker: str) -> str:
    """Normalize speaker label: keep Petitioner/Respondent/Unknown; else remove Jr./II/III/commas, use last name."""
    if speaker in (PETITIONER, RESPONDENT) or speaker in UNKNOWN_SPEAKER_LABELS:
        return speaker
    s = speaker.replace(",", "").strip()
    s = _strip_name_suffixes(s)
    if not s:
        return speaker
    parts = s.split()
    return parts[-1] if parts else s


def remove_fillers(text: str) -> str:
    """Remove (Inaudible), (Voice overlap), (Laughter), (Pause), (Applause), (Coughing)."""
    filler_patterns = [
        r"\s*\(Inaudible\)\s*",
        r"\s*\(Voice\s+overlap\)\s*",
        r"\s*\(Laughter\.?\)\s*",
        r"\s*\(Pause\.?\)\s*",
        r"\s*\(Applause\.?\)\s*",
        r"\s*\(Coughing\.?\)\s*",
        r"\s*\(Inaudible\.\)\s*",
    ]
    for pat in filler_patterns:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    return text


def collapse_repeated_speaker(turns: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Merge consecutive turns by the same speaker into one turn."""
    if not turns:
        return []
    out = []
    cur_speaker, cur_text = turns[0]
    for speaker, text in turns[1:]:
        if speaker == cur_speaker and cur_text and text:
            cur_text = cur_text.rstrip() + " " + text.lstrip()
        else:
            out.append((cur_speaker, cur_text))
            cur_speaker, cur_text = speaker, text
    out.append((cur_speaker, cur_text))
    return out


def parse_transcript(content: str) -> tuple[list[tuple[str, str]], str]:
    """Parse Speaker\\nText\\n... format; return (list of (speaker, text), footer after '---')."""
    lines = content.splitlines()
    footer_lines = []
    seen_dash = False
    main_lines = []
    for line in lines:
        if line.strip() == "---":
            seen_dash = True
            continue
        if seen_dash:
            footer_lines.append(line)
        else:
            main_lines.append(line)

    turns = []
    i = 0
    current_speaker = None
    current_text_lines = []

    while i < len(main_lines):
        line = main_lines[i]
        if line in KNOWN_SPEAKERS:
            if current_speaker is not None:
                text = "\n".join(current_text_lines).strip()
                if text or current_speaker:
                    turns.append((current_speaker, text))
            current_speaker = line
            current_text_lines = []
            i += 1
            continue
        if current_speaker is None:
            current_speaker = ""
        current_text_lines.append(line)
        i += 1

    if current_speaker is not None:
        text = "\n".join(current_text_lines).strip()
        turns.append((current_speaker, text))

    footer = "\n".join(footer_lines).strip() if footer_lines else ""
    return turns, footer


def serialize_transcript(turns: list[tuple[str, str]], footer: str) -> str:
    """Serialize turns back to Speaker\\nText format with optional ---\\nfooter."""
    parts = []
    for speaker, text in turns:
        parts.append(speaker)
        if text:
            parts.append(text)
    out = "\n".join(parts)
    if footer:
        out = out.rstrip() + "\n\n---\n" + footer
    return out + "\n" if not out.endswith("\n") else out


def clean_turn_text(text: str) -> str:
    """Apply filler removal and format_text to one turn's text."""
    text = remove_fillers(text)
    text = format_text(text)
    return text


def clean_transcript(
    content: str,
    normalize_speakers: bool = True,
    merge_same_speaker: bool = False,
) -> str:
    """Full cleaning: parse -> clean text (+ optional speaker norm + optional merge) -> serialize."""
    turns, footer = parse_transcript(content)
    cleaned = []
    for speaker, text in turns:
        if normalize_speakers and speaker:
            speaker = format_speaker(speaker)
        cleaned.append((speaker, clean_turn_text(text)))
    if merge_same_speaker:
        cleaned = collapse_repeated_speaker(cleaned)
    return serialize_transcript(cleaned, footer)


def transcript_has_justice_speaker(content: str) -> bool:
    """Return True if the transcript has at least one turn where the speaker is a justice (known last name)."""
    turns, _ = parse_transcript(content)
    for speaker, _ in turns:
        if speaker in KNOWN_JUSTICE_LAST_NAMES:
            return True
    return False


def remove_transcripts_without_justice_decisions(
    dir_path: str | Path,
) -> list[Path]:
    """
    Remove transcript .txt files that contain no justice decisions (no turn has a justice as speaker).
    Returns the list of file paths that were removed.
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        return []
    removed = []
    for path in dir_path.glob("*.txt"):
        try:
            content = path.read_text(encoding="utf-8")
            if not transcript_has_justice_speaker(content):
                path.unlink()
                removed.append(path)
        except Exception:
            continue
    return removed


# ---------------------------------------------------------------------------
# Pipeline and main
# ---------------------------------------------------------------------------

def process_case(
    convo,
    case_id: str,
    normalize_speakers: bool = True,
    merge_same_speaker: bool = False,
) -> str | None:
    """Produce cleaned transcript for one case: convo -> raw -> clean; prepend SCDB header + decided date.

    Returns None if the case has no SCDB match (caller should skip it).
    """
    scdb_header = format_scdb_header(str(case_id))
    if scdb_header is None:
        return None

    raw = convo_to_raw_transcript(convo)
    body = clean_transcript(
        raw,
        normalize_speakers=normalize_speakers,
        merge_same_speaker=merge_same_speaker,
    )
    legal = get_case_legal_info_from_corpus(str(case_id))
    decided_date = (legal or {}).get("decided_date")

    # Build header: SCDB metadata, then decided date, then transcript
    header = scdb_header
    if decided_date:
        header += f"\nDECIDED: {decided_date}"
    body = f"{header}\n\n{body}"
    return body


def load_corpus(max_utterances: int | None = None) -> Corpus:
    """Download Supreme Court corpus and load (optionally limited by utterance count)."""
    path = download("supreme-corpus")
    if max_utterances:
        return Corpus(
            filename=path,
            utterance_start_index=0,
            utterance_end_index=max_utterances - 1,
        )
    return Corpus(filename=path)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Export Supreme Court cases as cleaned transcripts (decided date + clean last names)."
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="case_transcripts_cleaned",
        help="Output directory (default: case_transcripts_cleaned)",
    )
    parser.add_argument(
        "--no-normalize-speakers",
        action="store_true",
        help="Do not normalize speaker labels (e.g. keep 'Jr.')",
    )
    parser.add_argument(
        "--merge-same-speaker",
        action="store_true",
        help="Merge consecutive turns by the same speaker",
    )
    parser.add_argument(
        "--keep-no-justice",
        action="store_true",
        help="Keep transcript files that have no justice as speaker (default is to remove them)",
    )
    args = parser.parse_args()

    print("Ensuring ConvoKit cases.jsonl is available...", file=sys.stderr)
    download_cases_jsonl()

    if MAX_UTTERANCES:
        print(f"Loading corpus (first {MAX_UTTERANCES} utterances)...", file=sys.stderr)
    else:
        print("Loading full corpus...", file=sys.stderr)
    corpus = load_corpus(MAX_UTTERANCES)

    convo_ids = corpus.get_conversation_ids()
    if not convo_ids:
        print("No conversations in corpus.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(convo_ids)} cleaned cases to {out_dir}/", file=sys.stderr)

    normalize_speakers = not args.no_normalize_speakers
    merge_same_speaker = args.merge_same_speaker
    seen_case_ids = set()
    skipped_no_scdb = 0

    for i, convo_id in tqdm(enumerate(convo_ids, 1), total=len(convo_ids), desc="Processing cases"):
        convo = corpus.get_conversation(convo_id)
        case_id = dict(convo.meta).get("case_id", convo_id)
        name = safe_filename(case_id)
        if name in seen_case_ids:
            name = f"{name}_{safe_filename(convo_id)}"
        seen_case_ids.add(name)
        out_path = out_dir / f"{name}.txt"
        try:
            text = process_case(
                convo,
                case_id,
                normalize_speakers=normalize_speakers,
                merge_same_speaker=merge_same_speaker,
            )
            if text is None:
                skipped_no_scdb += 1
                continue
            out_path.write_text(text, encoding="utf-8")
        except Exception as e:
            print(f"  Skip {case_id}: {e}", file=sys.stderr)
            continue
        if i % 100 == 0 or i == len(convo_ids):
            print(f"  {i}/{len(convo_ids)}", file=sys.stderr)

    if skipped_no_scdb:
        print(f"Skipped {skipped_no_scdb} case(s) with no SCDB match.", file=sys.stderr)

    if not args.keep_no_justice:
        removed = remove_transcripts_without_justice_decisions(out_dir)
        if removed:
            print(f"Removed {len(removed)} transcript(s) with no justice decisions.", file=sys.stderr)
            for p in removed[:10]:
                print(f"  {p.name}", file=sys.stderr)
            if len(removed) > 10:
                print(f"  ... and {len(removed) - 10} more", file=sys.stderr)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
