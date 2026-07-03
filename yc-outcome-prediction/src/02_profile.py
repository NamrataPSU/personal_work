#!/usr/bin/env python3
"""02_profile.py — Profile the raw YC companies dataset.

Reads data/raw/yc_companies_all.json and writes:
  - reports/01_data_profile.md   (human-readable profile)
  - data/raw/field_summary.csv   (machine-readable field summary)

No feature engineering or labeling — profiling only.
"""
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data" / "raw" / "yc_companies_all.json"
REPORT = PROJECT_ROOT / "reports" / "01_data_profile.md"
FIELD_CSV = PROJECT_ROOT / "data" / "raw" / "field_summary.csv"

# Manual annotation of each field's likely role in modeling.
FIELD_ROLES = {
    "id": "identifier",
    "name": "identifier",
    "slug": "identifier",
    "former_names": "feature-candidate (weak; pivot signal)",
    "small_logo_thumb_url": "identifier/URL",
    "website": "identifier/URL (presence/liveness could be outcome-adjacent)",
    "all_locations": "feature-candidate (location at listing; mostly stable)",
    "long_description": "feature-candidate (text; describes CURRENT product — mild leakage risk)",
    "one_liner": "feature-candidate (text; same caveat)",
    "team_size": "LEAKY/outcome-adjacent (CURRENT headcount reflects outcome, not starting state)",
    "industry": "feature-candidate",
    "subindustry": "feature-candidate",
    "launched_at": "feature-candidate (launch timestamp; relative to batch = early-traction signal)",
    "tags": "feature-candidate",
    "tags_highlighted": "identifier/derived (UI artifact of tags)",
    "top_company": "LEAKY (YC's post-hoc success designation)",
    "isHiring": "LEAKY/outcome-adjacent (current hiring status)",
    "nonprofit": "feature-candidate",
    "batch": "feature-candidate (cohort/vintage; also defines eval splits)",
    "status": "TARGET (outcome label source)",
    "industries": "feature-candidate (list form of industry+subindustry)",
    "regions": "feature-candidate (includes Remote flags)",
    "stage": "LEAKY/outcome-adjacent (CURRENT funding stage = outcome proxy)",
    "app_video_public": "feature-candidate (weak; disclosure choice)",
    "demo_day_video_public": "feature-candidate (weak; disclosure choice)",
    "app_answers": "feature-candidate (application answers; but ~all null)",
    "question_answers": "identifier/flag (whether app_answers exist)",
    "url": "identifier/URL",
    "api": "identifier/URL",
}

SEASON_ORDER = {"Winter": 0, "Spring": 1, "Summer": 2, "Fall": 3}


def batch_year(batch: str):
    """Extract 4-digit year from batch names like 'Winter 2012', 'Summer 2025',
    or legacy short forms like 'W21'/'S05' if present. 'Unspecified' -> None."""
    if not isinstance(batch, str):
        return None
    m = re.search(r"(19|20)\d{2}", batch)
    if m:
        return int(m.group(0))
    m = re.match(r"^[WSF](\d{2})$", batch)
    if m:
        return 2000 + int(m.group(1))
    return None


def example_vals(s: pd.Series, k: int = 3) -> str:
    vals = []
    for v in s.dropna():
        r = repr(v)
        if r not in vals:
            vals.append(r[:60])
        if len(vals) >= k:
            break
    return "; ".join(vals)


def md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown table without the tabulate dependency."""
    df = df.reset_index(drop=True)
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = ["" if pd.isna(v) else str(v) for v in row]
        out.append("| " + " | ".join(c.replace("|", "\\|").replace("\n", " ") for c in cells) + " |")
    return "\n".join(out)


def main() -> None:
    records = json.loads(RAW.read_text())
    df = pd.DataFrame(records)
    n = len(df)
    lines = ["# YC Companies Dataset — Data Profile",
             "",
             f"Source: `data/raw/yc_companies_all.json` (see `data/raw/PROVENANCE.md`). "
             f"Profiled with `src/02_profile.py`.",
             "",
             f"**Companies: {n:,}** | **Fields: {df.shape[1]}** | "
             f"Duplicate `id`s: {df['id'].duplicated().sum()} | "
             f"Duplicate `slug`s: {df['slug'].duplicated().sum()} | "
             f"Duplicate `name`s: {df['name'].duplicated().sum()}",
             ""]

    # ---------- Field summary ----------
    rows = []
    for col in df.columns:
        s = df[col]
        is_null = s.isna()
        # empty strings / empty lists counted separately
        try:
            n_empty_str = int((s == "").sum())
        except (TypeError, ValueError):
            n_empty_str = 0
        n_empty_list = int(s.apply(lambda v: isinstance(v, list) and len(v) == 0).sum())
        listy = s.apply(lambda v: isinstance(v, (list, dict))).any()
        n_unique = s.astype(str).nunique() if listy else s.nunique(dropna=True)
        pytypes = sorted({type(v).__name__ for v in s.dropna().head(2000)})
        rows.append({
            "field": col,
            "dtype": f"{s.dtype} ({'/'.join(pytypes) or 'all-null'})",
            "n_missing": int(is_null.sum()),
            "pct_missing": round(100 * is_null.mean(), 2),
            "n_empty_string": n_empty_str,
            "n_empty_list": n_empty_list,
            "n_unique": int(n_unique),
            "example_values": example_vals(s),
            "role": FIELD_ROLES.get(col, "UNANNOTATED"),
        })
    field_summary = pd.DataFrame(rows)
    field_summary.to_csv(FIELD_CSV, index=False)

    lines += ["## 1. Fields, types, missingness", "",
              md_table(field_summary[["field", "dtype", "n_missing", "pct_missing",
                                      "n_empty_string", "n_empty_list", "n_unique"]]),
              "", f"Machine-readable copy: `data/raw/field_summary.csv`.", ""]

    # ---------- Status ----------
    status_counts = df["status"].value_counts(dropna=False)
    st = status_counts.rename_axis("status").reset_index(name="count")
    st["pct"] = (100 * st["count"] / n).round(2)
    lines += ["## 2. Outcome variable: `status`", "", md_table(st), ""]

    # ---------- Batch coverage ----------
    df["batch_year"] = df["batch"].apply(batch_year)
    df["batch_season"] = df["batch"].str.extract(r"^(Winter|Spring|Summer|Fall)")[0]
    batch_fmt = df["batch"].value_counts()
    yr = df.groupby("batch_year", dropna=False).size().rename("companies").reset_index()
    lines += ["## 3. Batch coverage", "",
              f"Distinct batch labels: {df['batch'].nunique()}. "
              f"Year range: {int(df['batch_year'].min())}–{int(df['batch_year'].max())} "
              f"(rows with unparseable batch year: {int(df['batch_year'].isna().sum())}, "
              f"labels: {sorted(df.loc[df['batch_year'].isna(), 'batch'].unique().tolist())}).",
              "", "### Companies per batch", "",
              md_table(batch_fmt.rename_axis("batch").reset_index(name="count")), "",
              "### Companies per year", "", md_table(yr), ""]

    # ---------- Status by batch year ----------
    ct = pd.crosstab(df["batch_year"], df["status"], dropna=False)
    ct["total"] = ct.sum(axis=1)
    for c in [c for c in ct.columns if c != "total"]:
        ct[f"{c}_pct"] = (100 * ct[c] / ct["total"]).round(1)
    lines += ["## 4. `status` by batch year", "",
              md_table(ct.reset_index()), "",
              "_Note: recent batches are overwhelmingly Active simply because outcomes "
              "have not had time to resolve — right-censoring. Labeling agent must handle this._", ""]

    # ---------- Key distributions ----------
    ts = df["team_size"].describe()
    tsq = df["team_size"].quantile([0.5, 0.9, 0.99])
    lines += ["## 5. Key field distributions", "",
              "### team_size (WARNING: current headcount — leaky as a feature)", "",
              f"missing: {df['team_size'].isna().sum()} | zero: {(df['team_size'] == 0).sum()} | "
              f"mean: {ts['mean']:.1f} | median: {tsq[0.5]:.0f} | p90: {tsq[0.9]:.0f} | "
              f"p99: {tsq[0.99]:.0f} | max: {ts['max']:.0f}", ""]

    def vc_section(title, series, top=15):
        v = series.value_counts().head(top)
        return [f"### {title} (top {min(top, len(v))} of {series.nunique()} unique)", "",
                md_table(v.rename_axis(title.split()[0]).reset_index(name="count")), ""]

    lines += vc_section("industry", df["industry"])
    lines += vc_section("subindustry", df["subindustry"], 20)
    tags_flat = pd.Series([t for lst in df["tags"] if isinstance(lst, list) for t in lst])
    regions_flat = pd.Series([r for lst in df["regions"] if isinstance(lst, list) for r in lst])
    lines += vc_section("tags (flattened)", tags_flat, 20)
    lines += vc_section("regions (flattened)", regions_flat, 20)
    lines += vc_section("stage (WARNING: current stage — leaky)", df["stage"])
    lines += vc_section("all_locations (top values)", df["all_locations"], 15)

    # booleans
    bools = ["top_company", "isHiring", "nonprofit", "app_video_public",
             "demo_day_video_public", "question_answers"]
    brows = [{"field": b, "true": int((df[b] == True).sum()),  # noqa: E712
              "false": int((df[b] == False).sum()),  # noqa: E712
              "missing": int(df[b].isna().sum())} for b in bools]
    lines += ["### Boolean flags", "", md_table(pd.DataFrame(brows)), ""]

    # launched_at sanity
    la = pd.to_datetime(df["launched_at"], unit="s", errors="coerce")
    lines += ["### launched_at (unix timestamp)", "",
              f"missing: {df['launched_at'].isna().sum()} | "
              f"min: {la.min()} | max: {la.max()} | "
              f"pre-2005 (suspicious): {(la < '2005-01-01').sum()}", ""]

    # ---------- Feature vs leaky annotation ----------
    lines += ["## 6. Field roles for modeling (feature vs leaky vs identifier)", "",
              md_table(field_summary[["field", "role"]]), "",
              "**Founder-level data: NOT present.** No founder names, counts, bios, "
              "prior companies, or education fields exist in this dataset. `app_answers` "
              f"is null for {int(df['app_answers'].isna().sum())}/{n} rows. Founder features "
              "would require a separate source (e.g., per-company pages or YC's site).", ""]

    # ---------- Quirks ----------
    former = int(df["former_names"].apply(lambda v: isinstance(v, list) and len(v) > 0).sum())
    dup_names = df[df["name"].duplicated(keep=False)]["name"].nunique()
    lines += ["## 7. Data quirks", "",
              f"- Duplicate ids: {df['id'].duplicated().sum()}; duplicate slugs: "
              f"{df['slug'].duplicated().sum()}; names shared by >1 company: {dup_names}.",
              f"- Batch label formats present: {sorted(set(df['batch'].str.extract(chr(94) + '([A-Za-z]+)')[0].dropna()))} "
              f"+ year (long form only in this dump); 'Unspecified' batch rows: "
              f"{int((df['batch'] == 'Unspecified').sum())}.",
              f"- Empty strings vs nulls: `one_liner` empty-string rows: {int((df['one_liner'] == '').sum())}, "
              f"`long_description` empty: {int((df['long_description'] == '').sum())}, "
              f"`website` empty: {int((df['website'] == '').sum())}, "
              f"`all_locations` empty: {int((df['all_locations'] == '').sum())} — "
              "missing text is encoded as '' not null.",
              f"- team_size: {int(df['team_size'].isna().sum())} null and "
              f"{int((df['team_size'] == 0).sum())} zero values (zero for many dead companies).",
              f"- Companies with former_names: {former} (~half the dataset; mostly legal-name "
              "variants, not true pivots — treat with care).",
              f"- launched_at min is {la.min().date()} even though batches go back to 2005 — "
              "it records when the company was ADDED/LAUNCHED on the YC directory, not founding "
              "date; unreliable as a company-age feature for pre-2010 batches.",
              ""]

    REPORT.write_text("\n".join(lines))
    print(f"Wrote {REPORT} and {FIELD_CSV}")


if __name__ == "__main__":
    main()
