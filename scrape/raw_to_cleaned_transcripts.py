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
DEFAULT_OUTPUT_DIR = os.path.join(DATA_DIR, 'case_transcripts_cleaned')


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
    """Build JUSTICE VOTES and OUTCOME from basic.json case entry."""
    votes = case.get('votes') or []
    lines = ["JUSTICE VOTES:"]
    if not votes:
        lines.append("(none in metadata)")
    else:
        for v in votes:
            if not isinstance(v, dict):
                continue
            name = (v.get('name') or '').strip()
            vote = (v.get('vote') or '').strip()
            if name:
                # basic.json has majority/minority; ConvoKit has Petitioner/Respondent
                lines.append(f"{name}: {vote}")

    majority = (case.get('majority') or '').strip()
    outcome = f"OUTCOME: {majority}." if majority else "OUTCOME: Unknown."
    return "\n".join(lines) + "\n\n" + outcome


def serialize_cleaned(turns: list[tuple[str, str]], footer: str) -> str:
    """Same format as cleaned_transcript_generator: Speaker\\nText then ---\\nfooter."""
    parts = []
    for speaker, text in turns:
        parts.append(speaker)
        if text:
            parts.append(text)
    out = "\n".join(parts)
    if footer:
        out = out.rstrip() + "\n\n---\n" + footer
    return out + "\n" if not out.endswith("\n") else out


def convert_one(raw_path: str, case: dict, normalize_speakers: bool = True) -> str:
    """
    Read scraped JSON, clean turns, build footer from case, return cleaned text.
    """
    with open(raw_path, 'r', encoding='utf-8') as f:
        statements = json.load(f)
    if not isinstance(statements, list):
        statements = []

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

    footer = build_footer(case)
    return serialize_cleaned(turns, footer)


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
            text = convert_one(str(raw_path), case, normalize_speakers=normalize)
            name = case.get('name') or base
            out_name = safe_filename(name) + ".txt"
            out_path = out_dir / out_name
            # avoid overwriting different case with same safe name
            if out_path.exists() and out_path.stat().st_size > 0:
                existing = out_path.read_text(encoding='utf-8')
                if existing.strip() != text.strip():
                    out_name = f"{safe_filename(name)}_{base}.txt"
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
