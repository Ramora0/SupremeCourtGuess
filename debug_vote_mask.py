"""Debug: inspect vote_mask positions and what tokens they correspond to."""
import glob, os
from transformers import AutoTokenizer

DATA_DIR = "case_transcripts_cleaned"
VOTES_DELIMITER = "\n---\nJUSTICE VOTES:\n"
MAX_SEQ_LENGTH = 16384

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)

pet_first = tokenizer.encode(" Petitioner", add_special_tokens=False)[0]
res_first = tokenizer.encode(" Respondent", add_special_tokens=False)[0]
vote_tokens = {pet_first, res_first}
print(f"Vote token IDs: pet_first={pet_first}, res_first={res_first}")
print(f"  pet decodes to: {tokenizer.decode([pet_first])!r}")
print(f"  res decodes to: {tokenizer.decode([res_first])!r}")

# Check a few files
paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))[:5]
for path in paths:
    with open(path) as f:
        text = f.read()
    if VOTES_DELIMITER not in text:
        continue

    idx = text.index(VOTES_DELIMITER) + len(VOTES_DELIMITER)
    prompt = text[:idx]
    completion = text[idx:].strip()

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False) + [tokenizer.eos_token_id]

    # Left-truncate prompt
    max_prompt_len = MAX_SEQ_LENGTH - len(completion_ids)
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids

    # Find vote positions
    vote_positions = [i for i, lb in enumerate(labels) if lb in vote_tokens]

    # Also check: do vote tokens appear in PROMPT portion of input_ids?
    prompt_vote_positions = [i for i, tid in enumerate(input_ids[:len(prompt_ids)]) if tid in vote_tokens]

    print(f"\n{'='*60}")
    print(f"File: {os.path.basename(path)}")
    print(f"Prompt tokens: {len(prompt_ids)}, Completion tokens: {len(completion_ids)}")
    print(f"Vote positions in LABELS (completion only): {len(vote_positions)}")
    print(f"Vote token IDs in PROMPT input_ids: {len(prompt_vote_positions)}")
    print(f"\nCompletion text:\n{completion[:500]}")
    print(f"\nVote positions and surrounding context:")
    for pos in vote_positions:
        # Show tokens around this position
        start = max(0, pos - 3)
        end = min(len(input_ids), pos + 3)
        context_ids = input_ids[start:end]
        context_labels = labels[start:end]
        context_tokens = [tokenizer.decode([tid]) for tid in context_ids]
        vote_idx_in_context = pos - start
        print(f"  pos={pos} label={labels[pos]} token={tokenizer.decode([input_ids[pos]])!r}")
        print(f"    context: {context_tokens}")
        print(f"    label context: {context_labels}")

    # KEY CHECK: what does the model see at position pos-1 (the prediction position)?
    # With the shift, logits[pos-1] predicts labels[pos]
    # At position pos-1, via teacher forcing, the model sees input_ids[0:pos]
    # So it sees input_ids[pos-1] as the last token before predicting the vote
    print(f"\n  For FIRST vote at pos={vote_positions[0]}:")
    prev_pos = vote_positions[0] - 1
    prev_tokens = [tokenizer.decode([input_ids[i]]) for i in range(max(0, prev_pos-5), prev_pos+1)]
    print(f"    Tokens leading up to first vote: {prev_tokens}")
    print(f"    The model predicts: {tokenizer.decode([input_ids[vote_positions[0]]])!r}")
