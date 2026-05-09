"""
02_assemble_prompts.py
======================
Read the ground-truth CSV and emit the full SD prompt list.

Configuration (matches decisions made in conversation):
- 5 paraphrased templates × 20 images each = 100 per category
- Singular form for Fiske plural groups
- Drop "Unemployed" from He occupations
- Total: 66 categories × 100 = 6,600 prompts

Output
------
outputs/prompts.jsonl  - one JSON object per (category, template, sample_idx)
outputs/prompts_summary.csv  - human-readable overview

Usage
-----
    python 02_assemble_prompts.py --input groundtruth_fiske_he.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 5 paraphrased templates. Each uses {article} and {group} as placeholders.
# Articles: "a" / "an" auto-selected. Use "" if the category already has its
# own determiner (e.g., "the elderly").
# ---------------------------------------------------------------------------
TEMPLATES = [
    "A portrait photo of {article} {group}, high quality, neutral background.",
    "A photo of {article} {group}.",
    "A portrait of {article} {group}.",
    "A headshot of {article} {group}, photographic.",
    "A picture of {article} {group}, realistic.",
]

IMAGES_PER_TEMPLATE = 20  # 5 templates × 20 = 100 per category

# ---------------------------------------------------------------------------
# Singularization map for the Fiske plural groups in your CSV.
# Add/remove as needed if your CSV uses different wording.
# ---------------------------------------------------------------------------
SINGULARIZE = {
    "rich people":          "rich person",
    "feminists":            "feminist",
    "black professionals":  "black professional",
    "elderly people":       "elderly person",
    "housewives":           "housewife",
    "welfare recipients":   "welfare recipient",
}

# Categories to exclude entirely.
EXCLUDE = {"unemployed"}


def normalize_category(raw: str) -> str | None:
    """
    Apply singularization and exclusion rules.
    Returns None if the category should be dropped.
    """
    key = raw.strip().lower()
    if key in EXCLUDE:
        return None
    if key in SINGULARIZE:
        return SINGULARIZE[key]
    return key


def article_for(noun: str) -> str:
    """Pick 'a' / 'an' / '' for a singular noun phrase."""
    first = noun.strip().split()[0].lower()
    if first in {"a", "an", "the"}:
        return ""
    return "an" if first[0] in "aeiou" else "a"


def render(template: str, category: str) -> str:
    """Fill in template, fix double spaces from empty articles."""
    art = article_for(category)
    text = template.format(article=art, group=category)
    return " ".join(text.split())  # collapse whitespace


def deterministic_seed(category: str, template_idx: int, sample_idx: int) -> int:
    """
    Stable seed per (category, template, sample). Reproducible across runs.
    Uses a simple hash mod 2^31 (positive 32-bit int, safe for SD pipelines).
    """
    key = f"{category}|{template_idx}|{sample_idx}"
    h = 1469598103934665603  # FNV-1a 64-bit offset basis
    for ch in key.encode("utf-8"):
        h ^= ch
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h & 0x7FFFFFFF


def build_prompts(input_csv: Path, output_jsonl: Path, summary_csv: Path) -> None:
    with input_csv.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    n_dropped = 0
    n_singularized = 0
    prompts = []
    summary_rows = []

    for row in rows:
        raw = row["category"].strip()
        norm = normalize_category(raw)
        if norm is None:
            n_dropped += 1
            print(f"  drop: {raw}")
            continue
        if norm != raw.lower():
            n_singularized += 1
            print(f"  singularize: {raw} -> {norm}")

        for t_idx, tpl in enumerate(TEMPLATES):
            text = render(tpl, norm)
            for s_idx in range(IMAGES_PER_TEMPLATE):
                prompts.append({
                    "category": norm,
                    "category_original": raw,
                    "kind": row.get("kind", ""),
                    "source": row.get("source", ""),
                    "template_idx": t_idx,
                    "template": tpl,
                    "sample_idx": s_idx,
                    "prompt": text,
                    "seed": deterministic_seed(norm, t_idx, s_idx),
                })

        summary_rows.append({
            "category": norm,
            "category_original": raw,
            "kind": row.get("kind", ""),
            "source": row.get("source", ""),
            "n_prompts": len(TEMPLATES) * IMAGES_PER_TEMPLATE,
            "example_prompt": render(TEMPLATES[0], norm),
        })

    with output_jsonl.open("w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    n_categories = len(summary_rows)
    print()
    print(f"Wrote {len(prompts)} prompts across {n_categories} categories.")
    print(f"  templates: {len(TEMPLATES)}")
    print(f"  images per template: {IMAGES_PER_TEMPLATE}")
    print(f"  images per category: {len(TEMPLATES) * IMAGES_PER_TEMPLATE}")
    print(f"  dropped: {n_dropped}")
    print(f"  singularized: {n_singularized}")
    print(f"Output: {output_jsonl}")
    print(f"Summary: {summary_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/groundtruth_fiske_he.csv",
                    help="Path to ground-truth CSV with 'category' column.")
    ap.add_argument("--output", default=str(OUT / "prompts.jsonl"))
    ap.add_argument("--summary", default=str(OUT / "prompts_summary.csv"))
    args = ap.parse_args()

    build_prompts(Path(args.input), Path(args.output), Path(args.summary))


if __name__ == "__main__":
    main()