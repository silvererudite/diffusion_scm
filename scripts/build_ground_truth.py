"""
02_build_groundtruth_csv.py
===========================
Builds the schema for the human-rated W/C ground-truth table that you'll
compare CLIP-projected ratings against.

The CATEGORY list below is taken from:
- Fiske et al. (2002), Tables 1-4, samples 1-6, J. Pers. Soc. Psychol., 82(6).
- He, Kang, Tse & Toh (2019), Table 1 + Appendix A, J. Vocational Behavior, 115.

The actual W/C MEAN values are in the published tables of those papers.
This script does NOT contain those numbers — you must transcribe them from
the source PDFs. The csv it writes has empty 'warmth' and 'competence'
columns ready to be filled in.

Why no auto-scraping: the values appear in typeset PDF tables (Fiske: 23
groups × 4 dimensions; He: 60 occupations × 2 dimensions). Reliable
extraction requires either (a) finding an OSF supplement with the raw
numbers, which neither paper provides, or (b) OCR — both of which introduce
errors a thesis can't carry.

Output
------
groundtruth_template.csv  (~83 rows, columns to fill in)

Usage
-----
    python 02_build_groundtruth_csv.py
"""
from __future__ import annotations

import csv
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Fiske et al. (2002): identity / status groups
# Source: Tables 2-4 of the paper. Sample sizes vary by group; use the
# pooled means reported in Sample 6 (the validation sample, N=571) where
# possible. Warmth and competence are 1-5 scales.
# ---------------------------------------------------------------------------
FISKE_GROUPS = [
    "rich people", "wealthy people", "businesswomen", "Asians",
    "Jewish people", "feminists", "British people", "Northerners",
    "Americans", "Black professionals", "students", "young people",
    "middle-class people", "white people", "Christians", "Irish people",
    "Italians", "Hispanic people", "Black people",
    "elderly people", "disabled people", "housewives", "blind people",
    "retarded people",  # original term used in Fiske 2002 — modernise to "people with intellectual disabilities" if appropriate for your write-up
    "homeless people", "welfare recipients", "poor people",
    "Arabs", "Turks", "migrant workers",
]


# ---------------------------------------------------------------------------
# He, Kang, Tse & Toh (2019): occupations
# Source: Appendix A. Warmth and competence are 1-7 scales.
# Note: He et al. used 60 occupations. Listed below in the order they appear
# in the appendix. Some occupations overlap conceptually with Fiske groups
# (e.g., "homemaker" ~ "housewife") — keep them separate, they got
# different ratings in different studies.
# ---------------------------------------------------------------------------
HE_OCCUPATIONS = [
    "accountant", "actor", "architect", "artist", "athlete",
    "bank teller", "bartender", "bus driver", "carpenter", "cashier",
    "CEO", "chef", "computer programmer", "construction worker", "customer service representative",
    "dancer", "dental hygienist", "dentist", "doctor", "electrician",
    "elementary school teacher", "engineer", "factory worker", "farmer", "financial analyst",
    "firefighter", "flight attendant", "graphic designer", "hairdresser", "hotel receptionist",
    "housekeeper", "human resources manager", "investment banker", "janitor", "judge",
    "lawyer", "librarian", "machinist", "maid", "manager",
    "mechanic", "musician", "nurse", "paralegal", "pharmacist",
    "photographer", "physical therapist", "plumber", "police officer", "politician",
    "professor", "psychologist", "real estate agent", "receptionist", "salesperson",
    "scientist", "secretary", "social worker", "software developer", "soldier",
    "surgeon", "taxi driver", "truck driver", "veterinarian", "waiter",
]


def build_template():
    rows = []
    for g in FISKE_GROUPS:
        rows.append({
            "category": g,
            "kind": "identity_group",
            "source": "Fiske2002",
            "scale_max": 5,
            "warmth_mean": "",
            "warmth_sd": "",
            "competence_mean": "",
            "competence_sd": "",
            "n_raters": "",
            "table_reference": "",
            "notes": "",
        })
    for o in HE_OCCUPATIONS:
        rows.append({
            "category": o,
            "kind": "occupation",
            "source": "He2019",
            "scale_max": 7,
            "warmth_mean": "",
            "warmth_sd": "",
            "competence_mean": "",
            "competence_sd": "",
            "n_raters": "",
            "table_reference": "",
            "notes": "",
        })
    return rows


def main():
    rows = build_template()
    path = OUT / "groundtruth_template.csv"
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")
    print()
    print("NEXT STEPS")
    print("----------")
    print("1. Open the Fiske et al. (2002) PDF. Tables 2, 3, and 4 contain the")
    print("   warmth and competence means per group. The cleanest validation")
    print("   sample is Sample 6 (Table 4, p.890). Transcribe means + SDs.")
    print()
    print("2. Open the He, Kang, Tse & Toh (2019) PDF. Appendix A (the table at")
    print("   the end) lists all 60 occupations with their warmth and competence")
    print("   means. Transcribe.")
    print()
    print("3. NORMALISE before comparing across sources. Fiske is on a 1-5 scale,")
    print("   He is on a 1-7 scale. Standard fix: z-score each rating column")
    print("   within its source before merging.")
    print()
    print("4. The 'category' column is what you'll use as {group} in your SD")
    print("   prompt template. Sanity-check the wording — singular noun phrases")
    print("   work better than plural for portrait generation.")


if __name__ == "__main__":
    main()