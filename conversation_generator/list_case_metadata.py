#!/usr/bin/env python3
"""
List all available metadata for one Supreme Court case from the ConvoKit corpus.
Use this to see what data exists and which ConvoKit APIs to use.

Case legal info (case name, citation, decision date) for corpus cases (1940–2019)
comes from ConvoKit's cases.jsonl (same dataset, separate file). See
conversation_generator/corpus_case_info.py. Facts and question presented are
not in the ConvoKit corpus or cases.jsonl.

Memory: ConvoKit loads the entire corpus into RAM (all cases, utterances with
full text, and speakers). To use less memory, set MAX_UTTERANCES (e.g. 50000)
to load only the first N lines of utterances.jsonl; you'll get a subset of
cases and lower RAM use.
"""

from convokit import Corpus, download
import json
import os

from corpus_case_info import get_case_legal_info_from_corpus

# Load only first N utterance lines to reduce memory (default: load full corpus)
MAX_UTTERANCES = int(os.environ.get("MAX_UTTERANCES", 0)) or None  # e.g. 50000

# ConvoKit APIs used:
#   Corpus(filename=...)           - load corpus from path
#   download("supreme-corpus")     - download and return path to corpus
#   corpus.get_conversation_ids()  - list of conversation (case) IDs
#   corpus.get_conversation(id)    - get one conversation
#   convo.id, convo.meta           - conversation id and metadata dict
#   convo.get_utterance_ids()      - list of utterance IDs in this conversation
#   convo.get_utterance(id)        - get one utterance
#   convo.get_utterances_dataframe() - pandas DataFrame of all utterances
#   utt.id, utt.speaker, utt.text, utt.meta, utt.reply_to, utt.timestamp
#   utt.conversation_id
#   corpus.get_speaker(id)         - get speaker object (if you have speaker id)
#   speaker.id, speaker.meta      - speaker id and metadata
#   corpus.get_speaker_ids()      - all speaker IDs in corpus


def safe_dump(obj, max_len=2000):
    """Serialize to JSON; truncate long strings for display."""
    try:
        s = json.dumps(obj, indent=2, default=str)
    except Exception:
        s = str(obj)
    if len(s) > max_len:
        s = s[:max_len] + "\n... (truncated)"
    return s


def main():
    path = download("supreme-corpus")

    # --- Exhaustive: what files exist in the corpus directory ---
    print("=" * 60)
    print("CORPUS DIRECTORY (files on disk)")
    print("=" * 60)
    print(f"Path: {path}\n")
    if os.path.isdir(path):
        for f in sorted(os.listdir(path)):
            fp = os.path.join(path, f)
            if os.path.isfile(fp):
                size = os.path.getsize(fp)
                size_mb = size / (1024 * 1024)
                print(f"  {f:<25} {size_mb:>8.1f} MB")
        # index.json describes the schema of meta/vectors
        index_path = os.path.join(path, "index.json")
        if os.path.isfile(index_path):
            print("\n--- index.json (schema of meta/vector keys) ---")
            with open(index_path) as f:
                index = json.load(f)
            for k, v in sorted(index.items()):
                if k == "version":
                    print(f"  {k}: {v}")
                elif isinstance(v, dict):
                    print(f"  {k}:")
                    for kk, vv in sorted(v.items()):
                        print(f"    {kk}: {vv}")
        # Note on info.*.jsonl (annotation/vector data)
        info_files = [f for f in os.listdir(path) if f.startswith("info.") and f.endswith(".jsonl")]
        if info_files:
            print("\n--- Annotation / vector files (info.*.jsonl) ---")
            print("  These are loaded as corpus/utterance vectors when using preload_vectors.")
            for f in sorted(info_files):
                name = f.replace("info.", "").replace(".jsonl", "")
                print(f"  {f}  ->  vector name: '{name}'")
            print("  Example: Corpus(filename=path, preload_vectors=['parsed', 'tokens', 'arcs'])")
    else:
        print("  (directory not found)")
    print()

    if MAX_UTTERANCES:
        print(f"Loading ConvoKit Supreme Court corpus (first {MAX_UTTERANCES} utterances, low-memory)...")
        corpus = Corpus(
            filename=path,
            utterance_start_index=0,
            utterance_end_index=MAX_UTTERANCES - 1,
        )
    else:
        print("Loading ConvoKit Supreme Court corpus (full; this can use a lot of RAM)...")
        corpus = Corpus(filename=path)

    convo_ids = corpus.get_conversation_ids()
    if not convo_ids:
        print("No conversations in corpus (did you use MAX_UTTERANCES with too small a value?).")
        return
    case_id = convo_ids[0]
    convo = corpus.get_conversation(case_id)

    # Corpus-level metadata (applies to whole dataset, not just this case)
    print("\n" + "=" * 60)
    print("CORPUS LEVEL (dataset-wide)")
    print("=" * 60)
    print("API: corpus.meta or corpus.get_meta()")
    if corpus.meta and len(corpus.meta) > 0:
        for k, v in sorted(corpus.meta.items()):
            print(f"  {k}: {safe_dump(v, max_len=800)}")
    else:
        print("  (empty)")
    if getattr(corpus, "vectors", None) and len(corpus.vectors) > 0:
        print(f"  vectors (matrix names): {list(corpus.vectors)}")

    print("\n" + "=" * 60)
    print("CONVERSATION (CASE) LEVEL")
    print("=" * 60)
    print(f"\nAPI: corpus.get_conversation_ids() -> list of case IDs")
    print(f"API: corpus.get_conversation(case_id) -> Conversation")
    print(f"\nConversation ID: {convo.id}")

    print("\n--- Conversation metadata (convo.meta) ---")
    if convo.meta:
        for k, v in sorted(convo.meta.items()):
            print(f"  {k}: {safe_dump(v, max_len=800)}")
    else:
        print("  (empty)")

    # Case-level legal info from ConvoKit cases.jsonl (same dataset as corpus, 1940–2019)
    convokit_case_id = (convo.meta or {}).get("case_id") or convo.id
    legal = get_case_legal_info_from_corpus(str(convokit_case_id))
    print("\n--- Case legal info (from ConvoKit cases.jsonl) ---")
    if legal:
        print("  Source: ConvoKit cases.jsonl (see corpus_case_info.py).")
        print(f"  title (case name): {safe_dump(legal.get('title'), max_len=400)}")
        print(f"  citation: {safe_dump(legal.get('citation'), max_len=200)}")
        print(f"  decided_date: {safe_dump(legal.get('decided_date'), max_len=100)}")
        print(f"  petitioner: {safe_dump(legal.get('petitioner'), max_len=200)}")
        print(f"  respondent: {safe_dump(legal.get('respondent'), max_len=200)}")
        if legal.get("url"):
            print(f"  url: {legal['url']}")
        print("  Note: facts and question presented are not in the ConvoKit dataset.")
    else:
        print("  Not found. Ensure data/convokit_cases.jsonl exists (run corpus_case_info.download_cases_jsonl()).")

    utt_ids = convo.get_utterance_ids()
    print(f"\nUtterance count: {len(utt_ids)}")
    print("API: convo.get_utterance_ids() -> list of utterance IDs")
    print("API: convo.get_utterance(utt_id) -> Utterance")

    # Collect all unique utterance meta keys
    all_utt_meta_keys = set()
    for uid in utt_ids:
        u = convo.get_utterance(uid)
        if u.meta:
            all_utt_meta_keys.update(u.meta.keys())

    print("\n--- All utterance.meta keys (across this case) ---")
    for k in sorted(all_utt_meta_keys):
        print(f"  {k}")

    # Full dump of first 3 utterances
    print("\n" + "=" * 60)
    print("UTTERANCE LEVEL (first 3 utterances, full metadata)")
    print("=" * 60)
    for i, utt_id in enumerate(utt_ids[:3], 1):
        utt = convo.get_utterance(utt_id)
        print(f"\n--- Utterance {i} (id={utt.id}) ---")
        print(f"  text (first 200 chars): {repr(utt.text[:200])}")
        print(f"  conversation_id: {utt.conversation_id}")
        if getattr(utt, "vectors", None) and utt.vectors:
            print(f"  vectors (annotation names): {utt.vectors}")
        print(f"  reply_to: {utt.reply_to}")
        print(f"  timestamp: {utt.timestamp}")
        if utt.speaker:
            print(f"  speaker.id: {utt.speaker.id}")
            print(f"  speaker.meta: {safe_dump(dict(utt.speaker.meta), max_len=600)}")
        else:
            print("  speaker: None")
        print("  utterance.meta:")
        if utt.meta:
            for k, v in sorted(utt.meta.items()):
                print(f"    {k}: {safe_dump(v, max_len=400)}")
        else:
            print("    (empty)")

    # Speaker-level: list unique speakers in this case and their meta
    speaker_ids = set()
    for uid in utt_ids:
        u = convo.get_utterance(uid)
        if u.speaker:
            speaker_ids.add(u.speaker.id)

    print("\n" + "=" * 60)
    print("SPEAKER LEVEL (speakers in this case)")
    print("=" * 60)
    print("API: corpus.get_speaker(speaker_id) or utt.speaker")
    for sid in sorted(speaker_ids)[:15]:
        try:
            speaker = corpus.get_speaker(sid) if hasattr(corpus, "get_speaker") else None
            if speaker is None:
                # get from first utterance that has this speaker
                for uid in utt_ids:
                    u = convo.get_utterance(uid)
                    if u.speaker and u.speaker.id == sid:
                        speaker = u.speaker
                        break
            if speaker:
                print(f"\n  Speaker id: {speaker.id}")
                if getattr(speaker, "vectors", None) and speaker.vectors:
                    print(f"    vectors: {speaker.vectors}")
                if speaker.meta:
                    for k, v in sorted(speaker.meta.items()):
                        print(f"    {k}: {safe_dump(v, max_len=300)}")
                else:
                    print("    (no meta)")
        except Exception as e:
            print(f"  {sid}: (error: {e})")
    if len(speaker_ids) > 15:
        print(f"  ... and {len(speaker_ids) - 15} more speakers")

    # DataFrame columns (handy for ML)
    print("\n" + "=" * 60)
    print("DATAFRAME API (convo.get_utterances_dataframe())")
    print("=" * 60)
    df = convo.get_utterances_dataframe()
    print("Columns:", list(df.columns))
    print("\nFirst row (sample):")
    if len(df) > 0:
        for col in df.columns:
            val = df[col].iloc[0]
            print(f"  {col}: {safe_dump(val, max_len=200)}")


if __name__ == "__main__":
    main()
