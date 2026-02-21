#!/usr/bin/env python3
"""
Clean case_transcripts to reduce token count while preserving meaning.

Applies:
- format_text: strip, collapse newlines/spaces, remove "word -- word" stutters
- format_speaker: normalize speaker labels (Jr., commas, last name only)
- remove_fillers: (Inaudible), (Voice overlap), (Laughter.), etc.
- collapse_repeated_speaker: optional merge of consecutive same-speaker turns

Usage:
  python transcript_cleaner_1.py [--in-place] [--output DIR] [transcripts_dir]
  Default: read from case_transcripts/, write to case_transcripts_cleaned/
  --in-place: overwrite files in place (use with care)
"""

import argparse
import os
import re
import sys
from pathlib import Path

# Known speaker labels: Petitioner, Respondent, Jr., and Supreme Court justice last names.
# Used to parse "Speaker\nText\nSpeaker\nText" format without splitting on text lines.
PETITIONER = "Petitioner"
RESPONDENT = "Respondent"
# Historical + current justice last names (through 2025) for speaker-line detection.
KNOWN_JUSTICE_LAST_NAMES = {
    "Alito", "Baldwin", "Barbour", "Barrett", "Black", "Blackmun", "Blair", "Blatchford",
    "Bradley", "Brandeis", "Brennan", "Brewer", "Breyer", "Brown", "Burger", "Burton",
    "Butler", "Campbell", "Cardozo", "Catron", "Chase", "Clark", "Clarke", "Clifford",
    "Curtis", "Daniel", "Davis", "Day", "Douglas", "Duvall", "Ellsworth", "Field",
    "Fortas", "Frankfurter", "Fuller", "Ginsburg", "Goldberg", "Gorsuch", "Gray",
    "Grier", "Harlan", "Holmes", "Hughes", "Hunt", "Iredell", "Jackson", "Jay",
    "Johnson", "Kagan", "Kavanaugh", "Kennedy", "Lamar", "Livingston", "Lurton",
    "Marshall", "Marshall", "Matthews", "McKenna", "McKinley", "McLean", "McReynolds",
    "Miller", "Minton", "Moody", "Moore", "Murphy", "Nelson", "O'Connor", "Paterson",
    "Peckham", "Pitney", "Powell", "Reed", "Rehnquist", "Roberts", "Rutledge",
    "Sanford", "Scalia", "Shiras", "Sotomayor", "Souter", "Stevens", "Stewart",
    "Stone", "Story", "Strong", "Sutherland", "Swayne", "Taft", "Taney", "Thomas",
    "Thompson", "Todd", "Trimble", "VanDevanter", "Vinson", "Waite", "Warren",
    "Washington", "Wayne", "White", "Whittaker", "Wilson", "Woodbury", "Woods",
}
KNOWN_SPEAKERS = {PETITIONER, RESPONDENT, "Jr."} | KNOWN_JUSTICE_LAST_NAMES


def sub_all(pattern: str, sub: str, text: str) -> str:
    """Apply re.sub repeatedly until the text stops changing (for stutter patterns)."""
    new_text = re.sub(pattern, sub, text, flags=re.IGNORECASE)
    while new_text != text:
        text = new_text
        new_text = re.sub(pattern, sub, text, flags=re.IGNORECASE)
    return text


def format_text(text: str) -> str:
    """
    Clean utterance text to reduce tokens (from scrape/format_transcripts.py).
    - Strip whitespace
    - Remove newlines within text
    - Collapse multiple spaces to one
    - Remove stutter: "word -- word" -> "word", "word1 word2 -- word1 word2" -> "word1 word2"
    """
    if not text or not text.strip():
        return text.strip()
    text = text.strip()
    text = sub_all(r"\n", " ", text)
    text = sub_all(r"  +", " ", text)
    # Word boundaries prevent matching substrings (e.g. "o -- o" in "to -- or")
    text = sub_all(r"\b(\w+)\s--\s\1\b", r"\1", text)
    text = sub_all(r"\b(\w+)\s(\w+)\s--\s\1\s\2\b", r"\1 \2", text)
    return text.strip()


def format_speaker(speaker: str) -> str:
    """
    Normalize speaker label (from scrape/format_transcripts.py).
    - Petitioner/Respondent unchanged.
    - Remove "Jr.", commas, strip; return last word (last name only) for names.
    """
    if speaker in (PETITIONER, RESPONDENT):
        return speaker
    s = speaker.replace("Jr.", "").replace(",", "").strip()
    if not s:
        return speaker
    parts = s.split()
    return parts[-1] if parts else s


def remove_fillers(text: str) -> str:
    """
    Remove or shorten common non-content parentheticals to save tokens.
    - (Inaudible) -> remove
    - (Voice overlap) / (Voice Overlap) -> remove
    - (Laughter.) / (Laughter) -> remove
    - (Pause.) / (pause) -> remove
    - (Applause.) -> remove
    - Multiple spaces left after removal are collapsed later by format_text
    """
    # Remove parenthetical fillers; keep content like (citation).
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
    """
    Merge consecutive turns by the same speaker into one turn (optional token save).
    Reduces "Speaker\nShort\nSpeaker\nMore" to "Speaker\nShort More".
    """
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
    """
    Parse transcript format: Speaker\nText\nSpeaker\nText...
    Returns (list of (speaker, text) turns, optional footer after '---').
    """
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
        # Continuation of current speaker's text
        if current_speaker is None:
            # File may start with text (e.g. case header); treat as unnamed speaker
            current_speaker = ""
        current_text_lines.append(line)
        i += 1

    if current_speaker is not None:
        text = "\n".join(current_text_lines).strip()
        turns.append((current_speaker, text))

    footer = "\n".join(footer_lines).strip() if footer_lines else ""
    return turns, footer


def serialize_transcript(turns: list[tuple[str, str]], footer: str) -> str:
    """Back to Speaker\nText\n format, with optional footer after ---."""
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
    """Apply all text cleanings to one turn's text."""
    text = remove_fillers(text)
    text = format_text(text)
    return text


def clean_transcript(content: str, normalize_speakers: bool = True, merge_same_speaker: bool = False) -> str:
    """
    Full cleaning pipeline: parse -> clean text + optional speaker norm + optional merge -> serialize.
    """
    turns, footer = parse_transcript(content)
    cleaned = []
    for speaker, text in turns:
        if normalize_speakers and speaker:
            speaker = format_speaker(speaker)
        cleaned.append((speaker, clean_turn_text(text)))
    if merge_same_speaker:
        cleaned = collapse_repeated_speaker(cleaned)
    return serialize_transcript(cleaned, footer)


def main():
    parser = argparse.ArgumentParser(description="Clean case_transcripts to reduce tokens.")
    parser.add_argument(
        "transcripts_dir",
        nargs="?",
        default="case_transcripts",
        help="Directory containing .txt transcripts (default: case_transcripts)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory (default: transcripts_dir + '_cleaned', or same as input if --in-place)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite files in place (ignores --output)",
    )
    parser.add_argument(
        "--no-normalize-speakers",
        action="store_true",
        help="Do not normalize speaker labels (e.g. keep 'Jr.' as-is)",
    )
    parser.add_argument(
        "--merge-same-speaker",
        action="store_true",
        help="Merge consecutive turns by the same speaker into one",
    )
    args = parser.parse_args()

    src_dir = Path(args.transcripts_dir)
    if not src_dir.is_dir():
        print(f"Error: not a directory: {src_dir}", file=sys.stderr)
        sys.exit(1)

    if args.in_place:
        out_dir = src_dir
    else:
        out_dir = Path(args.output or f"{src_dir}_cleaned")
        out_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(src_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files in {src_dir}", file=sys.stderr)
        sys.exit(0)

    normalize_speakers = not args.no_normalize_speakers
    merge_same_speaker = args.merge_same_speaker

    for path in txt_files:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            cleaned = clean_transcript(
                content,
                normalize_speakers=normalize_speakers,
                merge_same_speaker=merge_same_speaker,
            )
            out_path = out_dir / path.name
            out_path.write_text(cleaned, encoding="utf-8")
        except Exception as e:
            print(f"Skip {path.name}: {e}", file=sys.stderr)

    print(f"Cleaned {len(txt_files)} file(s) -> {out_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
