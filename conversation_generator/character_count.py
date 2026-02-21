#!/usr/bin/env python3
"""
Print character count for each transcript file in a folder.

Usage:
  python3 character_count.py [folder_name]
  Default folder: case_transcripts
"""

import sys
from pathlib import Path


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "case_transcripts"
    path = Path(folder)
    if not path.is_dir():
        print(f"Error: not a directory: {path}", file=sys.stderr)
        sys.exit(1)

    files = sorted(path.glob("*.txt"))
    if not files:
        print(f"No .txt files in {path}", file=sys.stderr)
        sys.exit(0)

    total = 0
    for f in files:
        try:
            n = f.read_text(encoding="utf-8", errors="replace").__len__()
        except Exception as e:
            print(f"{f.name}: error - {e}", file=sys.stderr)
            continue
        total += n
        print(f"{n:>10}  {f.name}")
    print(f"{total:>10}  TOTAL ({len(files)} files)")


if __name__ == "__main__":
    main()
