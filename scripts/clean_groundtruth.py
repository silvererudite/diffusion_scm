"""
01_clean_groundtruth.py
-----------------------
Clean the groundtruth CSV. Two issues:

  1. Fiske2002 rows have a one-column shift: the 'warmth_sd' column actually
     contains the competence_mean, etc. He2019 rows are correctly aligned.
  2. Fiske is on a 1-5 raw scale; He is on a centered z-scale (~ -2 to +2).
     We z-score each source independently so they can be analyzed together.

Outputs:
  groundtruth_clean.csv  with columns:
    category, category_key, kind, source, scale_max,
    warmth_raw, competence_raw,           # in original units
    warmth_z, competence_z,               # z-scored within source
    notes
"""

import pandas as pd
from pathlib import Path

raw = pd.read_csv("groundtruth_fiske_he.csv")

# -------- 1. Pull warmth & competence per source --------
fiske_mask = raw["source"] == "Fiske2002"
he_mask    = raw["source"] == "He2019"

# Fiske: columns are shifted by one after warmth_mean
#   warmth_mean    -> warmth   (correct)
#   warmth_sd      -> competence_mean  (shifted!)
#   competence_mean-> n_raters string
fiske = pd.DataFrame({
    "category":       raw.loc[fiske_mask, "category"],
    "kind":           raw.loc[fiske_mask, "kind"],
    "source":         raw.loc[fiske_mask, "source"],
    "scale_max":      raw.loc[fiske_mask, "scale_max"],
    "warmth_raw":     raw.loc[fiske_mask, "warmth_mean"].astype(float),
    "competence_raw": raw.loc[fiske_mask, "warmth_sd"].astype(float),  # shifted!
})

# He: columns are correctly aligned
he = pd.DataFrame({
    "category":       raw.loc[he_mask, "category"],
    "kind":           raw.loc[he_mask, "kind"],
    "source":         raw.loc[he_mask, "source"],
    "scale_max":      raw.loc[he_mask, "scale_max"],
    "warmth_raw":     raw.loc[he_mask, "warmth_mean"].astype(float),
    "competence_raw": pd.to_numeric(raw.loc[he_mask, "competence_mean"], errors="coerce"),
})

# -------- 2. Z-score within source --------
for df in (fiske, he):
    df["warmth_z"]     = (df["warmth_raw"]     - df["warmth_raw"].mean())     / df["warmth_raw"].std()
    df["competence_z"] = (df["competence_raw"] - df["competence_raw"].mean()) / df["competence_raw"].std()

clean = pd.concat([fiske, he], ignore_index=True)

# -------- 3. Build join key (singularize Fiske plurals, lowercase) --------
PLURAL_TO_SINGULAR = {
    "rich people":         "rich person",
    "feminists":           "feminist",
    "black professionals": "black professional",
    "elderly people":      "elderly person",
    "housewives":          "housewife",
    "welfare recipients":  "welfare recipient",
}
def normalize(name: str) -> str:
    s = str(name).strip().lower()
    return PLURAL_TO_SINGULAR.get(s, s)

clean["category_key"] = clean["category"].apply(normalize)

assert clean["category_key"].duplicated().sum() == 0, "Duplicate join keys!"
assert clean["competence_raw"].isna().sum() == 0, "Missing competence values!"

clean.to_csv("groundtruth_clean.csv", index=False)
print(f"Wrote {len(clean)} rows ({fiske_mask.sum()} Fiske + {he_mask.sum()} He)")
print(f"Warmth z-range:     {clean['warmth_z'].min():.2f}  to  {clean['warmth_z'].max():.2f}")
print(f"Competence z-range: {clean['competence_z'].min():.2f}  to  {clean['competence_z'].max():.2f}")
print("\nSanity (Fiske):")
print(clean[clean.source=="Fiske2002"][["category","warmth_raw","competence_raw"]])