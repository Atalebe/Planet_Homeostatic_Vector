# scripts/check_ps_duplicates.py
# Usage:
#   .venv/bin/python scripts/check_ps_duplicates.py \
#       --ps data/raw/exoplanets/PS_2026.02.23_02.08.36.csv \
#       --tier4 data/derived/tier4/tier4_enriched.csv \
#       --planet "LHS 1140 b" \
#       --out data/derived/tier5/ps_selection_diagnostic_lhs1140b.csv
#
# What it does:
# 1) pulls all PS duplicate rows for the planet
# 2) computes a "distance to Tier4 baseline" score
# 3) selects closest_to_tier4 row
# 4) writes a diagnostic CSV + prints selected row with ref fields

import argparse
import json
import math
import numpy as np
import pandas as pd


NUM_COLS = [
    "pl_rade", "pl_bmasse", "pl_orbeccen", "pl_eqt", "pl_insol", "st_age",
    "pl_radeerr1", "pl_radeerr2",
    "pl_bmasseerr1", "pl_bmasseerr2",
    "pl_orbeccenerr1", "pl_orbeccenerr2",
    "pl_eqterr1", "pl_eqterr2",
    "pl_insolerr1", "pl_insolerr2",
    "st_ageerr1", "st_ageerr2",
    "sy_dist",
    "sy_gaiamag", "sy_tmag", "sy_kmag", "sy_jmag", "sy_hmag", "sy_vmag",
    "disc_year",
]
REF_COLS = [
    "pl_refname", "sy_refname", "disc_refname", "pl_pubdate", "disc_pubdate", "releasedate",
    "disc_facility", "disc_telescope", "disc_instrument", "discoverymethod",
]


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def safe_float(x):
    try:
        if x is None:
            return np.nan
        x = float(x)
        return x
    except Exception:
        return np.nan


def baseline_from_tier4(tier4_csv: str, planet_id: str) -> dict:
    df = pd.read_csv(tier4_csv, low_memory=False)
    m = df["planet_id"].astype(str).str.strip().eq(planet_id.strip())
    if not m.any():
        raise SystemExit(f"Planet '{planet_id}' not found in {tier4_csv}")
    row = df.loc[m].iloc[0]
    keys = ["pl_rade", "pl_bmasse", "pl_orbeccen", "pl_eqt", "pl_insol", "st_age"]
    base = {}
    for k in keys:
        base[k] = safe_float(row.get(k))
    return base


def dist_score(ps_row: pd.Series, base: dict) -> float:
    """
    L2 distance in a normalized space:
    - uses fractional residuals where baseline is finite and nonzero
    - uses absolute residuals when baseline is ~0
    - ignores fields missing on either side
    """
    keys = ["pl_rade", "pl_bmasse", "pl_orbeccen", "pl_eqt", "pl_insol", "st_age"]
    acc = 0.0
    n = 0
    for k in keys:
        xb = base.get(k, np.nan)
        xp = safe_float(ps_row.get(k, np.nan))
        if not np.isfinite(xb) or not np.isfinite(xp):
            continue
        denom = abs(xb) if abs(xb) > 1e-12 else 1.0
        r = (xp - xb) / denom
        acc += r * r
        n += 1
    if n == 0:
        return np.inf
    return math.sqrt(acc / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", required=True, help="NASA PS CSV")
    ap.add_argument("--tier4", required=True, help="Tier4 enriched CSV")
    ap.add_argument("--planet", required=True, help="Planet name/id (must match pl_name and tier4 planet_id)")
    ap.add_argument("--out", default="", help="Write diagnostic CSV here")
    args = ap.parse_args()

    base = baseline_from_tier4(args.tier4, args.planet)

    ps = pd.read_csv(args.ps, comment="#", low_memory=False)
    ps["pl_name"] = ps["pl_name"].astype(str).str.strip()
    sub = ps.loc[ps["pl_name"].eq(args.planet.strip())].copy()
    if len(sub) == 0:
        raise SystemExit(f"No PS rows for '{args.planet}' in {args.ps}")

    # numeric coercion for relevant columns
    for c in NUM_COLS:
        if c in sub.columns:
            sub[c] = to_num(sub[c])

    # compute distance to baseline
    sub["dist_to_tier4"] = sub.apply(lambda r: dist_score(r, base), axis=1)

    # pick closest row
    sub_sorted = sub.sort_values(["dist_to_tier4"]).reset_index(drop=True)
    best = sub_sorted.iloc[0]

    # print baseline and selection
    print("\nTier4 baseline (used for selection):")
    print(json.dumps(base, indent=2))

    print(f"\nPS duplicates for {args.planet}: {len(sub_sorted)}")
    cols_show = ["dist_to_tier4"] + [c for c in ["pl_rade","pl_bmasse","pl_orbeccen","pl_eqt","pl_insol","st_age"] if c in sub_sorted.columns]
    print("\nTop rows by dist_to_tier4:")
    print(sub_sorted[cols_show].head(10).to_string(index=False))

    print("\nSelected PS row (closest_to_tier4):")
    keep = []
    for c in ["pl_name"] + cols_show + REF_COLS:
        if c in sub_sorted.columns:
            keep.append(c)
    print(sub_sorted.loc[0, keep].to_string())

    # write diagnostics
    if args.out:
        out = sub_sorted.copy()
        # keep a compact set first, but include all err columns if present
        base_keep = ["pl_name", "dist_to_tier4"] + [c for c in NUM_COLS if c in out.columns] + [c for c in REF_COLS if c in out.columns]
        out = out[base_keep]
        out.to_csv(args.out, index=False)
        print("\nWrote diagnostic CSV:", args.out)


if __name__ == "__main__":
    main()
