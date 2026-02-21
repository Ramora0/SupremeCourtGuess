"""Tokenization statistics for all transcript .txt files.

Uses the same Qwen2.5-7B tokenizer from lees-branch train.py to tokenize
every file, then displays a comprehensive matplotlib dashboard of statistics.
"""

import glob
import os
import sys
import time

import numpy as np
from transformers import AutoTokenizer

# ── Config (mirrors lees-branch train.py) ─────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-7B"
TOKENS_PER_10S = 16_000  # training throughput estimate

DATA_DIRS = [
    "case_transcripts",
    "data/transcripts",
]


# ── Collect files ─────────────────────────────────────────────────────────────


def collect_txt_files(dirs: list[str]) -> list[str]:
    """Gather all .txt file paths from the given directories."""
    paths = []
    for d in dirs:
        if not os.path.isdir(d):
            print(f"[WARN] Directory not found, skipping: {d}")
            continue
        found = sorted(glob.glob(os.path.join(d, "*.txt")))
        print(f"  Found {len(found):,} files in {d}/")
        paths.extend(found)
    if not paths:
        print("ERROR: No .txt files found in any directory.")
        sys.exit(1)
    return paths


# ── Tokenize ──────────────────────────────────────────────────────────────────


def tokenize_files(paths: list[str], tokenizer) -> dict:
    """Tokenize each file and return per-file token counts + metadata."""
    token_counts = []
    char_counts = []
    filenames = []
    skipped = 0

    for i, path in enumerate(paths):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  Tokenizing {i + 1:,}/{len(paths):,} ...")
        with open(path) as f:
            text = f.read()
        if not text.strip():
            skipped += 1
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_counts.append(len(ids))
        char_counts.append(len(text))
        filenames.append(os.path.basename(path))

    print(f"  Tokenized {len(token_counts):,} files ({skipped} empty/skipped)")
    return {
        "token_counts": np.array(token_counts),
        "char_counts": np.array(char_counts),
        "filenames": filenames,
    }


# ── Statistics ────────────────────────────────────────────────────────────────


def compute_stats(arr: np.ndarray) -> dict:
    """Compute comprehensive statistics on an array."""
    q = np.percentile(arr, [5, 10, 25, 50, 75, 90, 95, 99])
    return {
        "count": len(arr),
        "total": int(arr.sum()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "p5": float(q[0]),
        "p10": float(q[1]),
        "q1": float(q[2]),
        "q2": float(q[3]),
        "q3": float(q[4]),
        "p90": float(q[5]),
        "p95": float(q[6]),
        "p99": float(q[7]),
        "iqr": float(q[4] - q[2]),
        "range": int(arr.max() - arr.min()),
    }


def fmt_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


# ── Dashboard ─────────────────────────────────────────────────────────────────


def build_dashboard(stats: dict, token_counts: np.ndarray, char_counts: np.ndarray,
                    filenames: list[str]):
    """Build a massive matplotlib figure with all the stats."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    total_tokens = stats["total"]
    est_seconds = total_tokens / TOKENS_PER_10S * 10
    est_per_epoch = fmt_time(est_seconds)
    est_3_epochs = fmt_time(est_seconds * 3)

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(
        "Supreme Court Transcript — Tokenization Analysis\n"
        f"(Qwen2.5-7B tokenizer  ·  {stats['count']:,} files)",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    gs = GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Histogram of token counts ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0:2])
    n_bins = min(100, max(30, stats["count"] // 50))
    ax1.hist(token_counts, bins=n_bins, color="#4C72B0", edgecolor="black",
             linewidth=0.3, alpha=0.85)
    ax1.axvline(stats["mean"], color="red", ls="--", lw=1.5, label=f"Mean: {stats['mean']:,.0f}")
    ax1.axvline(stats["median"], color="orange", ls="--", lw=1.5, label=f"Median: {stats['median']:,.0f}")
    ax1.set_title("Token Count Distribution", fontweight="bold")
    ax1.set_xlabel("Tokens per file")
    ax1.set_ylabel("Count")
    ax1.legend(fontsize=8)

    # ── 2. Log-scale histogram ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2:4])
    log_bins = np.logspace(np.log10(max(1, token_counts.min())),
                           np.log10(token_counts.max()), n_bins)
    ax2.hist(token_counts, bins=log_bins, color="#55A868", edgecolor="black",
             linewidth=0.3, alpha=0.85)
    ax2.set_xscale("log")
    ax2.axvline(stats["mean"], color="red", ls="--", lw=1.5, label=f"Mean: {stats['mean']:,.0f}")
    ax2.axvline(stats["median"], color="orange", ls="--", lw=1.5, label=f"Median: {stats['median']:,.0f}")
    ax2.set_title("Token Count Distribution (Log Scale)", fontweight="bold")
    ax2.set_xlabel("Tokens per file (log)")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=8)

    # ── 3. Box plot ───────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    bp = ax3.boxplot(token_counts, vert=True, patch_artist=True,
                     boxprops=dict(facecolor="#C44E52", alpha=0.7),
                     medianprops=dict(color="black", lw=2))
    ax3.set_title("Box Plot", fontweight="bold")
    ax3.set_ylabel("Tokens")
    ax3.set_xticks([])

    # ── 4. Violin plot ────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    vp = ax4.violinplot(token_counts, showmeans=True, showmedians=True,
                        showextrema=True)
    for body in vp["bodies"]:
        body.set_facecolor("#8172B2")
        body.set_alpha(0.7)
    ax4.set_title("Violin Plot", fontweight="bold")
    ax4.set_ylabel("Tokens")
    ax4.set_xticks([])

    # ── 5. CDF ────────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2:4])
    sorted_tokens = np.sort(token_counts)
    cdf = np.arange(1, len(sorted_tokens) + 1) / len(sorted_tokens)
    ax5.plot(sorted_tokens, cdf, color="#4C72B0", lw=2)
    # Mark percentiles
    for pct, color, label in [
        (50, "orange", "P50"), (75, "green", "P75"),
        (90, "red", "P90"), (95, "purple", "P95"),
    ]:
        val = np.percentile(token_counts, pct)
        ax5.axvline(val, color=color, ls=":", lw=1.2, alpha=0.8)
        ax5.axhline(pct / 100, color=color, ls=":", lw=0.8, alpha=0.4)
        ax5.annotate(f"{label}: {val:,.0f}", xy=(val, pct / 100),
                     fontsize=7, color=color,
                     xytext=(5, 5), textcoords="offset points")
    ax5.set_title("Cumulative Distribution Function (CDF)", fontweight="bold")
    ax5.set_xlabel("Tokens")
    ax5.set_ylabel("Cumulative Proportion")
    ax5.grid(True, alpha=0.3)

    # ── 6. Tokens vs Characters scatter ───────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 0:2])
    ax6.scatter(char_counts, token_counts, s=3, alpha=0.3, color="#4C72B0")
    # Fit line
    z = np.polyfit(char_counts, token_counts, 1)
    p = np.poly1d(z)
    x_line = np.linspace(char_counts.min(), char_counts.max(), 100)
    ax6.plot(x_line, p(x_line), "r--", lw=1.5,
             label=f"Fit: {z[0]:.3f}x + {z[1]:.0f}")
    avg_ratio = token_counts.sum() / char_counts.sum()
    ax6.set_title(f"Tokens vs Characters  (avg ratio: {avg_ratio:.3f} tok/char)",
                  fontweight="bold")
    ax6.set_xlabel("Characters")
    ax6.set_ylabel("Tokens")
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    # ── 7. Top 20 longest files ───────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2:4])
    top_idx = np.argsort(token_counts)[::-1][:20]
    top_tokens = token_counts[top_idx]
    top_labels = [os.path.splitext(filenames[i])[0] for i in top_idx]
    bars = ax7.barh(range(len(top_tokens)), top_tokens, color="#DD8452",
                    edgecolor="black", linewidth=0.3)
    ax7.set_yticks(range(len(top_tokens)))
    ax7.set_yticklabels(top_labels, fontsize=8)
    ax7.invert_yaxis()
    ax7.set_title("Top 20 Longest Files (by tokens)", fontweight="bold")
    ax7.set_xlabel("Tokens")
    for i, (bar, val) in enumerate(zip(bars, top_tokens)):
        ax7.text(val + token_counts.max() * 0.01, i, f"{val:,}", va="center", fontsize=7)

    # ── 8. Percentile breakdown bar chart ─────────────────────────────────
    ax8 = fig.add_subplot(gs[3, 0:2])
    pct_labels = ["P5", "P10", "Q1\n(P25)", "Median\n(P50)", "Q3\n(P75)", "P90", "P95", "P99"]
    pct_values = [stats["p5"], stats["p10"], stats["q1"], stats["q2"],
                  stats["q3"], stats["p90"], stats["p95"], stats["p99"]]
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(pct_labels)))
    bars8 = ax8.bar(pct_labels, pct_values, color=colors, edgecolor="black", linewidth=0.3)
    for bar, val in zip(bars8, pct_values):
        ax8.text(bar.get_x() + bar.get_width() / 2, val + token_counts.max() * 0.01,
                 f"{val:,.0f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax8.set_title("Percentile / Quartile Breakdown", fontweight="bold")
    ax8.set_ylabel("Tokens")
    ax8.grid(axis="y", alpha=0.3)

    # ── 9. Summary statistics table ───────────────────────────────────────
    ax9 = fig.add_subplot(gs[3, 2:4])
    ax9.axis("off")

    table_data = [
        ["Total Files", f"{stats['count']:,}"],
        ["Total Tokens", f"{stats['total']:,}"],
        ["Mean", f"{stats['mean']:,.1f}"],
        ["Std Dev", f"{stats['std']:,.1f}"],
        ["Median", f"{stats['median']:,.1f}"],
        ["Min", f"{stats['min']:,}"],
        ["Max", f"{stats['max']:,}"],
        ["Range", f"{stats['range']:,}"],
        ["IQR (Q3-Q1)", f"{stats['iqr']:,.1f}"],
        ["Q1 (25th)", f"{stats['q1']:,.1f}"],
        ["Q3 (75th)", f"{stats['q3']:,.1f}"],
        ["P95", f"{stats['p95']:,.1f}"],
        ["P99", f"{stats['p99']:,.1f}"],
        ["", ""],
        ["Training Estimate", f"@ {TOKENS_PER_10S:,} tok/10s"],
        ["Est. 1 Epoch", est_per_epoch],
        ["Est. 3 Epochs", est_3_epochs],
        ["Tokens/Char Ratio", f"{token_counts.sum() / char_counts.sum():.4f}"],
    ]

    table = ax9.table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        cellLoc="center",
        loc="center",
        colWidths=[0.45, 0.45],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)

    # Style header
    for j in range(2):
        table[0, j].set_facecolor("#4C72B0")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(2):
            if i % 2 == 0:
                table[i, j].set_facecolor("#f0f0f0")
            else:
                table[i, j].set_facecolor("#ffffff")
    # Highlight training rows
    for i in range(15, 18):
        for j in range(2):
            table[i, j].set_facecolor("#fff3cd")

    ax9.set_title("Summary Statistics & Training Estimates", fontweight="bold",
                  pad=15)

    plt.savefig("tokenization_stats.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved figure to tokenization_stats.png")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  Tokenization Analysis — Supreme Court Transcripts")
    print("=" * 60)

    print(f"\nLoading tokenizer: {MODEL_NAME}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"  Tokenizer loaded in {time.time() - t0:.1f}s")
    print(f"  Vocab size: {tokenizer.vocab_size:,}")

    print(f"\nCollecting .txt files ...")
    paths = collect_txt_files(DATA_DIRS)
    print(f"  Total: {len(paths):,} files")

    print(f"\nTokenizing ...")
    t0 = time.time()
    result = tokenize_files(paths, tokenizer)
    elapsed = time.time() - t0
    print(f"  Finished in {elapsed:.1f}s "
          f"({len(paths) / elapsed:.0f} files/sec)")

    token_counts = result["token_counts"]
    char_counts = result["char_counts"]

    print(f"\nComputing statistics ...")
    stats = compute_stats(token_counts)

    # Print summary to terminal too
    print(f"\n{'─' * 50}")
    print(f"  Files:       {stats['count']:>10,}")
    print(f"  Total tokens:{stats['total']:>10,}")
    print(f"  Mean:        {stats['mean']:>10,.1f}")
    print(f"  Std:         {stats['std']:>10,.1f}")
    print(f"  Median:      {stats['median']:>10,.1f}")
    print(f"  Min:         {stats['min']:>10,}")
    print(f"  Max:         {stats['max']:>10,}")
    print(f"  Q1:          {stats['q1']:>10,.1f}")
    print(f"  Q3:          {stats['q3']:>10,.1f}")
    print(f"  IQR:         {stats['iqr']:>10,.1f}")
    print(f"  P95:         {stats['p95']:>10,.1f}")
    print(f"  P99:         {stats['p99']:>10,.1f}")
    total_tokens = stats["total"]
    est_sec = total_tokens / TOKENS_PER_10S * 10
    print(f"{'─' * 50}")
    print(f"  Training estimate @ {TOKENS_PER_10S:,} tok / 10s:")
    print(f"    1 epoch:   {fmt_time(est_sec)}")
    print(f"    3 epochs:  {fmt_time(est_sec * 3)}")
    print(f"{'─' * 50}")

    print(f"\nBuilding dashboard ...")
    build_dashboard(stats, token_counts, char_counts, result["filenames"])


if __name__ == "__main__":
    main()
