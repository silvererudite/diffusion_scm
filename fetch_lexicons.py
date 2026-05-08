"""
01_fetch_lexicons.py
====================
Pull the Nicolas, Bai & Fiske (2021) seed lexicon (with manual W/C
annotations from Fraser et al. 2022) and consolidate it into anchor word
lists for defining warmth and competence axes in CLIP space.

Sources
-------
1. github.com/katiefraser/computational-SCM (GPL-3.0)
   data/ contains the Nicolas seed lexicon with 3-annotator manual W/C labels.
2. github.com/gandalfnicolas/SADCAT
   Full SADCAT R package — fallback for the broader Sociability/Morality/
   Ability/Assertiveness dictionaries.

Usage
-----
    pip install pandas requests
    python 01_fetch_lexicons.py
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

FRASER_ZIP = "https://github.com/katiefraser/computational-SCM/archive/refs/heads/main.zip"
SADCAT_ZIP = "https://github.com/gandalfnicolas/SADCAT/archive/refs/heads/master.zip"

WARMTH_DIMS = {"sociability", "morality", "warmth"}
COMPETENCE_DIMS = {"ability", "assertiveness", "agency", "competence"}


def download_and_extract(url: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        zf.extractall(dest)
    children = [p for p in dest.iterdir() if p.is_dir()]
    if not children:
        raise RuntimeError(f"Empty zip at {url}")
    return children[0]


def find_csvs(root: Path) -> list[Path]:
    return sorted(root.rglob("*.csv"))


def inspect_csvs(label: str, paths: Iterable[Path]) -> None:
    print(f"\n--- {label} CSVs ---")
    for p in paths:
        try:
            df = pd.read_csv(p, nrows=3)
            cols = list(df.columns)
        except Exception as e:
            cols = f"<read failed: {e}>"
        print(f"  {p.name}  cols={cols}")


def _to_valence(v) -> int:
    if isinstance(v, (int, float)):
        return 1 if v > 0 else -1
    s = str(v).strip().lower()
    if s in {"pos", "positive", "+", "+1", "1", "high"}:
        return +1
    if s in {"neg", "negative", "-", "-1", "low"}:
        return -1
    return 0


def normalise_lexicon(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with columns: word, dimension, valence (+1/-1)."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    word_col = next(
        (c for c in df.columns if c in {"word", "term", "lexeme", "lemma"}),
        df.columns[0],
    )

    # Long format
    if {"dimension", "valence"}.issubset(df.columns):
        out = df.rename(columns={word_col: "word"})[["word", "dimension", "valence"]]
        out["valence"] = out["valence"].map(_to_valence)
        return out

    # Wide format: columns like "sociability_pos", "ability_neg", etc.
    rows = []
    for col in df.columns:
        if col == word_col:
            continue
        parts = col.replace("-", "_").split("_")
        dim_token = parts[0]
        pol_token = parts[-1] if len(parts) > 1 else "pos"
        if dim_token in WARMTH_DIMS or dim_token in COMPETENCE_DIMS:
            mask = df[col].fillna(0).astype(float) != 0
            sub = df.loc[mask, [word_col]].copy()
            sub.columns = ["word"]
            sub["dimension"] = dim_token
            sub["valence"] = +1 if pol_token.startswith(("p", "h")) else -1
            rows.append(sub)
    if not rows:
        return pd.DataFrame(columns=["word", "dimension", "valence"])
    return pd.concat(rows, ignore_index=True)


def consolidate(fraser_root: Path, sadcat_root: Path) -> pd.DataFrame:
    frames = []
    for repo_label, root in [("fraser", fraser_root), ("sadcat", sadcat_root)]:
        for csv in find_csvs(root):
            try:
                df = pd.read_csv(csv)
            except Exception:
                continue
            norm = normalise_lexicon(df)
            if len(norm) == 0:
                continue
            norm["source_repo"] = repo_label
            norm["source_file"] = csv.name
            frames.append(norm)

    if not frames:
        raise RuntimeError("No usable lexicon CSV found. Inspect raw_*/ folders.")

    combined = pd.concat(frames, ignore_index=True)
    combined["word"] = combined["word"].astype(str).str.strip().str.lower()
    combined = combined[combined["word"].str.match(r"^[a-z\-]{2,}$")]
    combined = combined.drop_duplicates(subset=["word", "dimension", "valence"])
    return combined


def split_anchors(combined: pd.DataFrame) -> dict[str, list[str]]:
    def get(dims: set[str], valence: int) -> list[str]:
        m = combined["dimension"].isin(dims) & (combined["valence"] == valence)
        return sorted(combined.loc[m, "word"].unique().tolist())

    return {
        "warmth_pos": get(WARMTH_DIMS, +1),
        "warmth_neg": get(WARMTH_DIMS, -1),
        "competence_pos": get(COMPETENCE_DIMS, +1),
        "competence_neg": get(COMPETENCE_DIMS, -1),
    }


def main() -> None:
    print("[1/3] Fetching repos ...")
    fraser_root = download_and_extract(FRASER_ZIP, OUT / "raw_fraser_data")
    sadcat_root = download_and_extract(SADCAT_ZIP, OUT / "raw_sadcat_data")

    print("\n[2/3] Inspecting CSVs ...")
    inspect_csvs("Fraser", find_csvs(fraser_root))
    inspect_csvs("SADCAT", find_csvs(sadcat_root))

    print("\n[3/3] Building anchor lists ...")
    combined = consolidate(fraser_root, sadcat_root)
    combined.to_csv(OUT / "anchors_combined.csv", index=False)

    anchors = split_anchors(combined)
    for key, words in anchors.items():
        path = OUT / f"{key}_anchors.txt"
        path.write_text("\n".join(words) + "\n")
        print(f"  {key:16s} -> {len(words):4d} words  ({path.name})")

    (OUT / "summary.json").write_text(
        json.dumps({k: len(v) for k, v in anchors.items()}, indent=2)
    )

    print("\nDone. Outputs in:", OUT)
    print("Next: open anchors_combined.csv and manually trim to ~30-50 highest-")
    print("quality words per pole. Prefer Fraser-annotated entries over auto-")
    print("expanded SADCAT — manual labels have better signal-to-noise.")


if __name__ == "__main__":
    sys.exit(main())