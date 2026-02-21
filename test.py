# supreme_convokit_inspect.py
# pip install convokit pandas

from convokit import Corpus, download
import json

def safe_preview(obj, n=1000):
    """Pretty-print helper that won't explode on non-serializable values."""
    try:
        s = json.dumps(obj, indent=2, default=str)
    except Exception:
        s = str(obj)
    if len(s) > n:
        s = s[:n] + "\n... (truncated)"
    return s

def main():
    print("Downloading / loading ConvoKit Supreme Court corpus...")
    corpus = Corpus(filename=download("supreme-corpus"))

    # Get all case IDs (ConvoKit conversations ~= cases here)
    convo_ids = corpus.get_conversation_ids()
    print(f"\nTotal conversations (cases): {len(convo_ids)}")

    # Pick one case
    case_id = convo_ids[0]
    convo = corpus.get_conversation(case_id)

    print("\n=== CASE / CONVERSATION OBJECT ===")
    print(f"conversation_id: {convo.id}")

    # Conversation metadata (varies by corpus)
    print("\nConversation metadata keys:")
    print(list(convo.meta.keys()))

    print("\nConversation metadata preview:")
    print(safe_preview(dict(convo.meta), n=2000))

    # Utterances in this case
    utt_ids = convo.get_utterance_ids()
    print(f"\nUtterance count in this case: {len(utt_ids)}")

    print("\n=== FIRST 5 UTTERANCES ===")
    for i, utt_id in enumerate(utt_ids[:5], 1):
        utt = convo.get_utterance(utt_id)
        print(f"\n--- Utterance {i} ---")
        print(f"id: {utt.id}")
        print(f"speaker.id: {utt.speaker.id if utt.speaker else None}")
        print(f"conversation_id: {utt.conversation_id}")
        print(f"reply_to: {utt.reply_to}")
        print(f"timestamp: {utt.timestamp}")
        print(f"text: {utt.text[:300]!r}")

        # metadata often contains useful Supreme Court-specific fields
        print(f"utterance.meta keys: {list(utt.meta.keys())[:20]}")
        if utt.meta:
            # print a tiny sample of meta
            meta_items = list(dict(utt.meta).items())[:5]
            print("utterance.meta sample:", safe_preview(dict(meta_items), n=1200))

    # Optional: dataframe view (handy for ML preprocessing)
    print("\n=== UTT DF HEAD ===")
    utt_df = convo.get_utterances_dataframe()
    print(utt_df.head(3).to_string())

if __name__ == "__main__":
    main()