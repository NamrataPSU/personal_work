"""Label definition for YC outcome prediction.

Parses batch -> (season, year) -> batch_date, computes company age as of
REFERENCE_DATE, and builds two label variants on the mature cohort
(batch_date <= MATURITY_CUTOFF, i.e. >= ~5 years of runway to resolve):

  label_strict   : 1 = Acquired|Public, 0 = Inactive, NaN = Active (excluded)
  label_survival : 1 = Acquired|Public|Active, 0 = Inactive

Rationale / limitations (see reports/02_labels_features.md):
  - Acquired includes acquihires and fire sales -- indistinguishable here.
  - Active at 5+ years may be a zombie or a thriving private company --
    indistinguishable here. Hence the two variants as a sensitivity pair.
  - Young cohorts (post mid-2021) are right-censored (~99% Active in 2025-26)
    and are excluded from the labeled population entirely.

Alternatives considered (not implemented): treating Active as positive only
above a team-size threshold (rejected: team_size is a current-state leaky
field); a longer 7-10y maturity window (rejected: halves the cohort);
survival-analysis framing with censoring (left to the modeling agent if
desired -- batch_date + status are sufficient to reconstruct it).

Outputs: data/processed/labels.csv and a per-year balance printout.
Importable: build_labels(df) is reused by 04_features.py.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "yc_companies_all.json"
OUT = ROOT / "data" / "processed" / "labels.csv"

REFERENCE_DATE = pd.Timestamp("2026-07-02")
MATURITY_YEARS = 5.0
# batch_date <= mid-2021 -> includes Winter 2021 (Jan) and Summer 2021 (Jun)
MATURITY_CUTOFF = pd.Timestamp("2021-07-02")

SEASON_MONTH = {"Winter": 1, "Spring": 4, "Summer": 6, "Fall": 9}


def parse_batch(batch: str):
    """'Summer 2021' -> ('Summer', 2021, Timestamp('2021-06-01')); else NaT."""
    parts = str(batch).strip().split()
    if len(parts) == 2 and parts[0] in SEASON_MONTH and parts[1].isdigit():
        season, year = parts[0], int(parts[1])
        return season, year, pd.Timestamp(date(year, SEASON_MONTH[season], 1))
    return None, np.nan, pd.NaT


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add batch parsing, age, maturity flag, and both label variants."""
    parsed = df["batch"].map(parse_batch)
    df = df.copy()
    df["batch_season"] = [p[0] for p in parsed]
    df["batch_year"] = [p[1] for p in parsed]
    df["batch_date"] = [p[2] for p in parsed]
    df["company_age_years"] = (REFERENCE_DATE - df["batch_date"]).dt.days / 365.25
    df["is_mature"] = df["batch_date"].notna() & (df["batch_date"] <= MATURITY_CUTOFF)

    success = df["status"].isin(["Acquired", "Public"])
    failure = df["status"].eq("Inactive")
    active = df["status"].eq("Active")

    df["label_strict"] = np.where(
        df["is_mature"] & success, 1.0,
        np.where(df["is_mature"] & failure, 0.0, np.nan),
    )
    df["label_survival"] = np.where(
        df["is_mature"] & (success | active), 1.0,
        np.where(df["is_mature"] & failure, 0.0, np.nan),
    )
    return df


def balance_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    sub = df[df[col].notna()]
    t = (
        sub.groupby("batch_year")[col]
        .agg(n="count", n_pos="sum")
        .assign(n_neg=lambda x: x["n"] - x["n_pos"],
                pos_rate=lambda x: (x["n_pos"] / x["n"]).round(3))
    )
    total = pd.DataFrame(
        {"n": [t["n"].sum()], "n_pos": [t["n_pos"].sum()],
         "n_neg": [t["n_neg"].sum()],
         "pos_rate": [round(t["n_pos"].sum() / t["n"].sum(), 3)]},
        index=["TOTAL"],
    )
    return pd.concat([t, total])


def main():
    df = pd.DataFrame(json.loads(RAW.read_text()))
    df = build_labels(df)

    unparsed = df[df["batch_date"].isna()]
    print(f"Rows: {len(df)} | unparseable batch rows excluded from labels: "
          f"{len(unparsed)} ({sorted(unparsed['batch'].unique())})")
    print(f"Mature cohort (batch_date <= {MATURITY_CUTOFF.date()}): "
          f"{int(df['is_mature'].sum())}")
    mature_active_excluded = int(
        (df["is_mature"] & df["status"].eq("Active")).sum())
    print(f"Mature Active companies (excluded from label_strict, positive in "
          f"label_survival): {mature_active_excluded}\n")

    for col in ["label_strict", "label_survival"]:
        print(f"=== {col} per batch year ===")
        print(balance_table(df, col).to_string(), "\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["id", "name", "batch", "batch_season", "batch_year", "batch_date",
            "company_age_years", "is_mature", "status",
            "label_strict", "label_survival"]
    df[cols].to_csv(OUT, index=False)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
