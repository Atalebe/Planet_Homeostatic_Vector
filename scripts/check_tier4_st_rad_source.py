# scripts/check_tier4_st_rad_source.py
# Usage:
#   .venv/bin/python scripts/check_tier4_st_rad_source.py \
#       --tier4 data/derived/tier4/tier4_enriched.csv \
#       --ps data/raw/exoplanets/PS_2026.02.23_02.08.36.csv \
#       --planet "LHS 1140 b"
#
# What it does:
# 1) shows st_rad used by Tier4
# 2) shows all PS duplicate st_rad values for the same planet (if present)
# This guards against silent mismatch between pipeline star radius and PS star radius.

import argparse
import pandas as pd

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier4", required=True)
    ap.add_argument("--ps", required=True)
    ap.add_argument("--planet", required=True)
    args = ap.parse_args()

    t4 = pd.read_csv(args.tier4, low_memory=False)
    m = t4["planet_id"].astype(str).str.strip().eq(args.planet.strip())
    if not m.any():
        raise SystemExit("Planet not found in tier4_enriched.")
    row = t4.loc[m].iloc[0]
    print("\nTier4 st_rad baseline:")
    print(" planet:", row["planet_id"])
    print(" st_rad (Tier4) =", row.get("st_rad"))

    ps = pd.read_csv(args.ps, comment="#", low_memory=False)
    ps["pl_name"] = ps["pl_name"].astype(str).str.strip()
    sub = ps.loc[ps["pl_name"].eq(args.planet.strip())].copy()
    print("\nPS duplicates:", len(sub))

    if "st_rad" in sub.columns:
        sub["st_rad"] = to_num(sub["st_rad"])
        print("\nPS st_rad values (unique):")
        vals = sorted([v for v in sub["st_rad"].dropna().unique().tolist()])
        print(vals[:50])
    else:
        print("\nPS has no st_rad column in this export (unexpected for PS, but possible for custom exports).")

if __name__ == "__main__":
    main()
