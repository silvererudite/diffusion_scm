"""
trim_anchors.py
===============
Takes anchors_combined.csv and prepares a trimming worksheet.
Outputs anchors_for_trimming.csv where you mark keep=1 or 0
in column 'keep', then re-run with --finalize to produce
the final four anchor txt files.
"""
import argparse
import pandas as pd
from pathlib import Path

OUT = Path("outputs")

def prepare():
    df = pd.read_csv(OUT / "anchors_combined.csv")
    # Prioritise Fraser entries; mark them keep=1 by default
    df["keep"] = (df["source_repo"] == "fraser").astype(int)
    # Sort: Fraser first, then by dimension/valence for easy scanning
    df = df.sort_values(
        by=["source_repo", "dimension", "valence", "word"],
        ascending=[True, True, False, True],
    )
    out = OUT / "anchors_for_trimming.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out} with {len(df)} rows.")
    print("Edit the 'keep' column (1=keep, 0=drop), aim for ~30-50 per pole.")
    print("Then run: python trim_anchors.py --finalize")

def finalize():
    df = pd.read_csv(OUT / "anchors_for_trimming.csv")
    kept = df[df["keep"] == 1]
    print("After trim:")
    for (dim, val), grp in kept.groupby(["dimension", "valence"]):
        pole = "pos" if val == 1 else "neg"
        print(f"  {dim} {pole}: {len(grp)} words")

    # Write final pooled lists
    WARMTH = {"sociability", "morality", "warmth"}
    COMPETENCE = {"ability", "assertiveness", "agency", "competence"}

    for name, dims, val in [
        ("warmth_pos_final", WARMTH, +1),
        ("warmth_neg_final", WARMTH, -1),
        ("competence_pos_final", COMPETENCE, +1),
        ("competence_neg_final", COMPETENCE, -1),
    ]:
        words = sorted(kept.loc[
            kept["dimension"].isin(dims) & (kept["valence"] == val), "word"
        ].unique())
        path = OUT / f"{name}.txt"
        path.write_text("\n".join(words) + "\n")
        print(f"  -> {path.name}: {len(words)} words")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    if args.finalize:
        finalize()
    else:
        prepare()