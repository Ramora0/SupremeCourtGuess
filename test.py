"""Debug tokenization of vote words to find correct token IDs for vote_mask."""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)

# How does the tokenizer split these in different contexts?
tests = [
    "Petitioner",
    "Respondent",
    " Petitioner",
    " Respondent",
    ": Petitioner",
    ": Respondent",
    "Earl Warren: Petitioner\n",
    "Earl Warren: Respondent\n",
    "Tom C. Clark: Petitioner\nOUTCOME: Petitioner won.",
]

print("=== Token IDs for each test string ===\n")
for text in tests:
    ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = [tokenizer.decode([i]) for i in ids]
    print(f"Text: {text!r}")
    print(f"  IDs:    {ids}")
    print(f"  Tokens: {tokens}")
    print()

# Now simulate what happens in the actual completion
print("=== Simulating actual completion tokenization ===\n")
completion = """Earl Warren: Respondent
Felix Frankfurter: Respondent
Hugo L. Black: Respondent
Tom C. Clark: Petitioner

OUTCOME: Respondent won."""

ids = tokenizer.encode(completion, add_special_tokens=False)
tokens = [tokenizer.decode([i]) for i in ids]
for i, (tid, tok) in enumerate(zip(ids, tokens)):
    print(f"  [{i:3d}] id={tid:6d}  token={tok!r}")

# Check what the standalone encode gives
print("\n=== Standalone encode ===")
pet_first = tokenizer.encode("Petitioner", add_special_tokens=False)[0]
res_first = tokenizer.encode("Respondent", add_special_tokens=False)[0]
print(f"pet_first={pet_first} ({tokenizer.decode([pet_first])!r})")
print(f"res_first={res_first} ({tokenizer.decode([res_first])!r})")

print(f"\nDo these IDs appear in the completion? pet_first: {pet_first in ids}, res_first: {res_first in ids}")
