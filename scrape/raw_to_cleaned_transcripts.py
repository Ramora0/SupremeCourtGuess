#!/usr/bin/env python3
"""
Convert scraped transcript JSON (data/raw_cases/*.json) to the same format as
case_transcripts_cleaned: Speaker\\nText per turn, then ---\\nJUSTICE VOTES\\nOUTCOME.

Uses basic.json to match raw files to cases and to build the footer (votes, outcome).
Output: one .txt per case in an output dir (default data/case_transcripts_cleaned).
"""
import json
import os
import re
from pathlib import Path

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW_CASES_DIR = os.path.join(DATA_DIR, 'raw_cases')
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'case_transcripts_cleaned')


def safe_filename(name: str) -> str:
    """Filesystem-safe case name (match conversation_generator)."""
    s = str(name).strip()
    s = re.sub(r'[<>:"/\\|?*]', '_', s)
    return s or "unknown"


def safe_filename_from_url(url: str) -> str:
    """Same as get_transcripts: URL -> base filename (no .json)."""
    s = url.replace('/', '-').strip()
    s = re.sub(r'[<>:"|?*]', '_', s)
    return s or "unknown"


def load_raw_statements(raw_path: str) -> list:
    """
    Load the list of {speaker, text} from a raw_cases JSON file.
    Supports both formats: a plain list, or a dict with 'statements' (from get_transcripts).
    """
    with open(raw_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and 'statements' in data:
        return data['statements'] if isinstance(data['statements'], list) else []
    return []


# ---------------------------------------------------------------------------
# Text/speaker cleaning (same logic as conversation_generator cleaned_transcript_generator)
# ---------------------------------------------------------------------------

def _sub_all(pattern: str, sub: str, text: str) -> str:
    while True:
        new_text = re.sub(pattern, sub, text, flags=re.IGNORECASE)
        if new_text == text:
            return text
        text = new_text


def format_text(text: str) -> str:
    """Collapse newlines/spaces, remove stutter patterns."""
    if not text or not text.strip():
        return text.strip()
    text = text.strip()
    text = _sub_all(r"\n", " ", text)
    text = _sub_all(r"  +", " ", text)
    text = _sub_all(r"\b(\w+)\s--\s\1\b", r"\1", text)
    text = _sub_all(r"\b(\w+)\s(\w+)\s--\s\1\s\2\b", r"\1 \2", text)
    return text.strip()


def format_speaker(speaker: str) -> str:
    """Normalize speaker: keep Petitioner/Respondent; else last name, no Jr./comma."""
    if speaker in ("Petitioner", "Respondent"):
        return speaker
    s = speaker.replace("Jr.", "").replace(",", "").strip()
    if not s:
        return speaker
    parts = s.split()
    return parts[-1] if parts else s


def remove_fillers(text: str) -> str:
    """Remove (Inaudible), (Voice overlap), etc."""
    patterns = [
        r"\s*\(Inaudible\)\s*", r"\s*\(Voice\s+overlap\)\s*",
        r"\s*\(Laughter\.?\)\s*", r"\s*\(Pause\.?\)\s*",
        r"\s*\(Applause\.?\)\s*", r"\s*\(Coughing\.?\)\s*",
        r"\s*\(Inaudible\.\)\s*",
    ]
    for pat in patterns:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    return text


def clean_turn_text(text: str) -> str:
    text = remove_fillers(text)
    return format_text(text)


def build_footer(case: dict) -> str:
    """Build JUSTICE VOTES and OUTCOME from case dict.
    Votes use Petitioner/Respondent side labels (matching ConvoKit format).
    Votes are shuffled randomly (matching ConvoKit generator behavior).
    """
    import random
    votes = case.get('votes') or []
    lines = ["JUSTICE VOTES:"]
    if not votes:
        lines.append("(none in metadata)")
    else:
        shuffled = list(votes)
        random.shuffle(shuffled)
        for v in shuffled:
            if not isinstance(v, dict):
                continue
            name = (v.get('name') or '').strip()
            # New format: votes have 'side' field (Petitioner/Respondent/Unknown)
            side = (v.get('side') or '').strip()
            if not side:
                # Fallback for old format with 'vote' field (majority/minority)
                side = (v.get('vote') or '').strip()
            if name:
                lines.append(f"{name}: {side}")

    win_side = case.get('win_side', -1)
    if win_side == 1:
        outcome = "OUTCOME: Petitioner won."
    elif win_side == 0:
        outcome = "OUTCOME: Respondent won."
    elif win_side == 2:
        outcome = "OUTCOME: Unclear."
    else:
        outcome = "OUTCOME: Unknown."
    return "\n".join(lines) + "\n\n" + outcome


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


def build_header(case: dict) -> str:
    """Build header with case context: year, facts, legal question."""
    parts = []
    year = case.get('year')
    if year:
        parts.append(f"YEAR: {year}")
    facts = (case.get('facts') or '').strip()
    if facts:
        parts.append(f"FACTS: {facts}")
    question = (case.get('question') or '').strip()
    if question:
        parts.append(f"QUESTION: {question}")
    return "\n\n".join(parts)


def serialize_cleaned(turns: list[tuple[str, str]], header: str, footer: str) -> str:
    """Header, then Speaker\\nText turns, then ---\\nfooter."""
    parts = []
    for speaker, text in turns:
        parts.append(speaker)
        if text:
            parts.append(text)
    out = "\n".join(parts)
    if header:
        out = header + "\n\n---\n" + out
    if footer:
        out = out.rstrip() + "\n\n---\n" + footer
    return out + "\n" if not out.endswith("\n") else out


def convert_one(raw_path: str, case: dict | None, normalize_speakers: bool = True, merge_same_speaker: bool = False) -> str:
    """
    Read scraped JSON, clean turns, build footer from case or embedded metadata.
    Raw file can be: (1) list of {speaker, text} — case must be provided;
    (2) dict with statements, name, votes, majority — case optional (used as fallback).
    """
    with open(raw_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        statements = data
        footer_case = case or {}
    else:
        statements = data.get('statements') if isinstance(data, dict) else []
        if not isinstance(statements, list):
            statements = []
        # Use embedded metadata when present; fall back to basic.json case
        if isinstance(data, dict) and (data.get('votes') is not None or data.get('win_side') is not None):
            footer_case = {
                'name': data.get('name') or (case.get('name') if case else ''),
                'votes': data.get('votes') or [],
                'win_side': data.get('win_side', case.get('win_side', -1) if case else -1),
            }
        else:
            footer_case = case or {}

    turns = []
    for st in statements:
        if not isinstance(st, dict):
            continue
        speaker = (st.get('speaker') or '').strip()
        text = (st.get('text') or '').strip()
        if normalize_speakers and speaker:
            speaker = format_speaker(speaker)
        text = clean_turn_text(text)
        if speaker or text:
            turns.append((speaker, text))

    if merge_same_speaker:
        turns = collapse_repeated_speaker(turns)

    header = build_header(case or {})
    footer = build_footer(footer_case)
    return serialize_cleaned(turns, header, footer)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert raw_cases JSON to cleaned .txt format.")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for .txt files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-normalize-speakers",
        action="store_true",
        help="Do not normalize speaker labels",
    )
    parser.add_argument(
        "--merge-same-speaker",
        action="store_true",
        help="Merge consecutive turns by the same speaker",
    )
    args = parser.parse_args()

    basic_path = os.path.join(DATA_DIR, 'basic.json')
    if not os.path.isfile(basic_path):
        print(f"basic.json not found at {basic_path}")
        return
    with open(basic_path, 'r') as f:
        raw = json.load(f)
    main_list = raw if isinstance(raw, list) else []

    raw_dir = Path(RAW_CASES_DIR)
    if not raw_dir.is_dir():
        print(f"Raw cases dir not found: {RAW_CASES_DIR}")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    normalize = not args.no_normalize_speakers
    merge = args.merge_same_speaker

    matched = 0
    written = 0
    for case in main_list:
        if not isinstance(case, dict):
            continue
        transcript_url = case.get('transcript_url')
        if not transcript_url:
            continue
        base = safe_filename_from_url(transcript_url)
        raw_path = raw_dir / f"{base}.json"
        if not raw_path.is_file():
            continue
        matched += 1
        try:
            text = convert_one(str(raw_path), case, normalize_speakers=normalize, merge_same_speaker=merge)
            year = case.get('year', '')
            # Extract docket number (case ID) from URL, e.g. "23-1039" from ".../cases/2024/23-1039"
            main_url = case.get('main_url') or ''
            case_id = main_url.rstrip('/').rsplit('/', 1)[-1] if main_url else ''
            case_id = safe_filename(case_id) if case_id else safe_filename(case.get('name') or base)
            out_name = f"{year}_{case_id}.txt"
            out_path = out_dir / out_name
            # avoid overwriting different case with same safe name
            if out_path.exists() and out_path.stat().st_size > 0:
                existing = out_path.read_text(encoding='utf-8')
                if existing.strip() != text.strip():
                    out_name = f"{year}_{case_id}_{base}.txt"
                    out_path = out_dir / out_name
            out_path.write_text(text, encoding='utf-8')
            written += 1
        except Exception as e:
            print(f"Error converting {case.get('name', base)}: {e}")

    print(f"Cases with transcript_url: {sum(1 for c in main_list if isinstance(c, dict) and c.get('transcript_url'))}")
    print(f"Matched raw files: {matched}")
    print(f"Written: {written} -> {out_dir}")


if __name__ == '__main__':
    main()
