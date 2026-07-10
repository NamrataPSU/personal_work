"""Feature engineering for YC outcome prediction.

Reads data/raw/yc_companies_all.json, applies labels from 03_labels.py, and
writes:
  data/processed/features.parquet     -- one row per company (all 5,999)
  data/processed/feature_manifest.csv -- column roles + leakage annotations
  data/processed/text_tfidf.npz       -- TF-IDF matrix (all rows; fit on
                                         mature cohort only), saved separately
                                         so the modeling agent can drop it
  data/processed/text_tfidf_vocab.json
  data/processed/README.md            -- written by hand alongside this script

Leakage policy: team_size / stage / isHiring / top_company are CURRENT-state
outcome proxies -- kept only as leaky_* diagnostic columns, never features.
launched_at floors at 2010 (directory-listing time) -> kept as
leaky_launched_at diagnostic only. one_liner/long_description describe the
CURRENT product (mild leakage) -- included but flagged.
"""

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

# import build_labels from 03_labels.py (module name starts with a digit)
_spec = importlib.util.spec_from_file_location("labels03", ROOT / "src" / "03_labels.py")
labels03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(labels03)

TOP_K_TAGS = 25
# canonicalize the overlapping AI tags into one merged tag
AI_TAGS = {"AI", "Artificial Intelligence", "Generative AI"}

BAY_AREA_PAT = re.compile(
    r"\b(?:San Francisco|Palo Alto|Mountain View|Menlo Park|Redwood City|"
    r"San Mateo|Sunnyvale|Santa Clara|San Jose|Cupertino|Oakland|Berkeley|"
    r"Fremont|Burlingame|South San Francisco|Hayward|San Carlos|Foster City|"
    r"Emeryville|Los Altos|Milpitas|Campbell)\b, CA", re.I)

META_REGIONS = {"Remote", "Partly Remote", "Fully Remote", "Unspecified"}


def clean_str(s: pd.Series) -> pd.Series:
    """Empty strings -> NaN; strip whitespace."""
    s = s.astype("object").str.strip()
    return s.replace("", np.nan)


def slugify(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")


def main():
    df = pd.DataFrame(json.loads((ROOT / "data" / "raw" / "yc_companies_all.json").read_text()))
    df = labels03.build_labels(df)
    PROC.mkdir(parents=True, exist_ok=True)

    out = df[["id", "name", "batch", "status", "batch_date", "batch_year",
              "company_age_years", "is_mature", "label_strict",
              "label_survival"]].copy()

    # --- batch / cohort ---
    out["batch_season"] = df["batch_season"]
    out["cohort_size"] = df.groupby("batch")["id"].transform("count")

    # --- industry ---
    out["industry"] = clean_str(df["industry"]).replace("Unspecified", np.nan)
    out["subindustry"] = clean_str(df["subindustry"]).replace("Unspecified", np.nan)
    out["n_industries"] = df["industries"].map(len)

    # --- tags (canonicalized) ---
    canon = df["tags"].map(lambda ts: sorted({"AI" if t in AI_TAGS else t for t in ts}))
    out["n_tags"] = df["tags"].map(len)
    freq = pd.Series([t for ts in canon for t in ts]).value_counts()
    top_tags = list(freq.head(TOP_K_TAGS).index)
    for t in top_tags:
        out[f"tag_{slugify(t)}"] = canon.map(lambda ts, t=t: int(t in ts)).astype("int8")

    # --- region / location ---
    regions = df["regions"]
    locs = clean_str(df["all_locations"])
    out["primary_region"] = regions.map(
        lambda rs: next((r for r in rs if r not in META_REGIONS), np.nan))
    out["is_us"] = regions.map(lambda rs: int("United States of America" in rs)).astype("int8")
    out["is_remote"] = regions.map(
        lambda rs: int(any(r in {"Remote", "Fully Remote", "Partly Remote"} for r in rs))).astype("int8")
    out["is_bay_area"] = locs.fillna("").str.contains(BAY_AREA_PAT).astype("int8")
    known_geo = regions.map(
        lambda rs: any(r not in META_REGIONS for r in rs))
    out["is_international"] = ((out["is_us"] == 0) & known_geo).astype("int8")
    out["has_location"] = locs.notna().astype("int8")

    # --- text-derived (current-day descriptions: mild leakage, see README) ---
    one_liner = clean_str(df["one_liner"])
    long_desc = clean_str(df["long_description"])
    out["one_liner_len"] = one_liner.str.len().fillna(0).astype(int)
    out["long_description_len"] = long_desc.str.len().fillna(0).astype(int)
    out["has_description"] = long_desc.notna().astype("int8")
    out["text_clean"] = (
        (one_liner.fillna("") + " " + long_desc.fillna(""))
        .str.replace(r"\s+", " ", regex=True).str.strip().replace("", np.nan))

    # --- simple flags (all set at/near batch time; see manifest) ---
    out["has_former_names"] = df["former_names"].map(lambda x: int(len(x) > 0)).astype("int8")
    out["nonprofit"] = df["nonprofit"].astype("int8")
    out["app_video_public"] = df["app_video_public"].astype("int8")
    out["demo_day_video_public"] = df["demo_day_video_public"].astype("int8")

    # --- leaky diagnostics (NOT features) ---
    out["leaky_team_size"] = df["team_size"]
    out["leaky_stage"] = df["stage"]
    out["leaky_is_hiring"] = df["isHiring"].astype("int8")
    out["leaky_top_company"] = df["top_company"].astype("boolean")
    out["leaky_launched_at"] = pd.to_datetime(df["launched_at"], unit="s")

    # --- TF-IDF: fit ONLY on mature cohort, transform everyone ---
    text = out["text_clean"].fillna("")
    vec = TfidfVectorizer(max_features=2000, min_df=5, stop_words="english",
                          sublinear_tf=True)
    vec.fit(text[out["is_mature"]])
    X = vec.transform(text)
    sparse.save_npz(PROC / "text_tfidf.npz", X)
    (PROC / "text_tfidf_vocab.json").write_text(json.dumps(
        {"row_order": "same as features.parquet (all 5999 rows, by id column)",
         "fit_population": "mature cohort only (is_mature == True)",
         "params": {"max_features": 2000, "min_df": 5,
                    "stop_words": "english", "sublinear_tf": True},
         "n_features": X.shape[1],
         "vocabulary": vec.get_feature_names_out().tolist()}, indent=1))

    # --- manifest ---
    def rows():
        meta = "batch-derived metadata; not a model feature"
        for c, d in [("id", "YC company id"), ("name", "company name"),
                     ("batch", "raw batch label"), ("status", "raw outcome (label source)"),
                     ("batch_date", meta), ("batch_year", meta),
                     ("company_age_years", "age as of 2026-07-02; " + meta),
                     ("is_mature", "batch_date <= 2021-07-02; defines labeled population")]:
            yield c, "metadata", d, "none", "identifier/bookkeeping"
        yield ("label_strict", "label", "1=Acquired|Public, 0=Inactive, NaN=Active/immature",
               "none", "target")
        yield ("label_survival", "label", "1=Acquired|Public|Active(mature), 0=Inactive",
               "none", "target")
        yield ("batch_season", "feature_categorical", "Winter/Spring/Summer/Fall",
               "none", "fixed at admission")
        yield ("cohort_size", "feature_numeric", "companies in same batch",
               "none", "fixed at admission; timing/competition signal")
        yield ("industry", "feature_categorical", "9 industries", "low",
               "could reflect a post-batch pivot but usually stable")
        yield ("subindustry", "feature_categorical", "59 subindustries", "low",
               "same as industry")
        yield ("n_industries", "feature_numeric", "len(industries list)", "low",
               "same as industry")
        yield ("n_tags", "feature_numeric", "raw tag count", "low",
               "tags curated over time but describe the business, not outcome")
        for t in top_tags:
            yield (f"tag_{slugify(t)}", "feature_numeric",
                   f"tag indicator: {t}" + (" (merged AI/Artificial Intelligence/Generative AI)"
                                            if t == "AI" else ""),
                   "low", "tags may be retagged post-hoc (esp. AI) but not outcome-derived")
        yield ("primary_region", "feature_categorical",
               "first non-Remote/meta region", "none", "fixed at admission")
        for c in ["is_us", "is_remote", "is_bay_area", "is_international", "has_location"]:
            yield (c, "feature_numeric", f"location flag ({c})", "low",
                   "location at listing; companies occasionally relocate")
        yield ("one_liner_len", "feature_numeric", "chars in one_liner", "medium",
               "one_liner describes CURRENT product; dead cos may have stale/empty text")
        yield ("long_description_len", "feature_numeric", "chars in long_description",
               "medium", "same; some descriptions literally say 'Acquired by X'")
        yield ("has_description", "feature_numeric", "long_description non-empty",
               "medium", "empty text correlates with defunct listings")
        yield ("text_clean", "feature_text", "one_liner + long_description, whitespace-normalized",
               "medium", "current-day text; modeling agent should ablate")
        yield ("has_former_names", "feature_numeric", "any former names listed", "low",
               "mostly legal-name variants; pivots add names post-batch")
        yield ("nonprofit", "feature_numeric", "nonprofit flag", "none", "fixed at admission")
        yield ("app_video_public", "feature_numeric", "application video public", "low",
               "video recorded pre-batch; publishing choice may come later")
        yield ("demo_day_video_public", "feature_numeric", "demo day video public", "low",
               "demo-day video is early-stage (end of batch); publishing choice may come later")
        yield ("leaky_team_size", "leaky_diagnostic", "CURRENT headcount", "high",
               "reflects outcome (0/NaN for dead cos, large for winners)")
        yield ("leaky_stage", "leaky_diagnostic", "CURRENT stage Early/Growth", "high",
               "outcome proxy")
        yield ("leaky_is_hiring", "leaky_diagnostic", "currently hiring", "high",
               "only live companies hire")
        yield ("leaky_top_company", "leaky_diagnostic", "YC 'top company' badge", "high",
               "post-hoc success designation")
        yield ("leaky_launched_at", "leaky_diagnostic",
               "directory listing/launch timestamp", "high",
               "floors at 2010 (directory age, not founding); also updated on relaunches")

    manifest = pd.DataFrame(rows(), columns=["column", "role", "description",
                                             "leakage_risk", "justification"])
    assert set(manifest["column"]) == set(out.columns), (
        set(manifest["column"]) ^ set(out.columns))
    manifest.to_csv(PROC / "feature_manifest.csv", index=False)

    out.to_parquet(PROC / "features.parquet", index=False)
    print(f"features.parquet: {out.shape[0]} rows x {out.shape[1]} cols")
    print(f"tfidf: {X.shape}, nnz={X.nnz}")
    print("top tags:", top_tags)
    print(manifest["role"].value_counts().to_string())


if __name__ == "__main__":
    main()
