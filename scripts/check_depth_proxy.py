# scripts/check_depth_proxy.py
# Usage:
#   .venv/bin/python scripts/check_depth_proxy.py \
#       --tier4 data/derived/tier4/tier4_enriched.csv \
#       --tier4-summary data/derived/tier4/summary.json \
#       --planet "LHS 1140 b"
#
# What it does:
# 1) recomputes the transit depth proxy (Rp/Rs)^2 from Tier4 columns
# 2) checks what Tier4 stored as tran_depth_proxy
# 3) prints the exact radii conversions used (Rsun -> Rearth)

import argparse
import json
import numpy as np
import pandas as pd

R_SUN_TO_R_EARTH = 109.076  # good enough for proxy work; keep consistent in pipeline

def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier4", required=True)
    ap.add_argument("--tier4-summary", default="")
    ap.add_argument("--planet", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.tier4, low_memory=False)
    m = df["planet_id"].astype(str).str.strip().eq(args.planet.strip())
    if not m.any():
        raise SystemExit(f"Planet '{args.planet}' not found in {args.tier4}")
    r = df.loc[m].iloc[0].copy()

    Rp = float(to_num(r.get("pl_rade")))
    Rs = float(to_num(r.get("st_rad")))  # Rsun
    stored = float(to_num(r.get("tran_depth_proxy")))

    Rs_rearth = Rs * R_SUN_TO_R_EARTH if np.isfinite(Rs) else np.nan
    depth = (Rp / Rs_rearth) ** 2 if (np.isfinite(Rp) and np.isfinite(Rs_rearth) and Rs_rearth > 0) else np.nan

    print("\nInputs from tier4_enriched:")
    print(f" planet_id       = {r.get('planet_id')}")
    print(f" pl_rade [Re]    = {Rp}")
    print(f" st_rad  [Rsun]  = {Rs}")
    print(f" st_rad  [Re]    = {Rs_rearth}")
    print(f" stored tran_depth_proxy = {stored}")

    print("\nRecomputed proxy:")
    print(f" depth_proxy = (Rp/Rs)^2 = {depth}")

    if np.isfinite(stored) and np.isfinite(depth):
        print("\nConsistency check:")
        print(" abs diff =", abs(stored - depth))
        print(" rel diff =", abs(stored - depth) / max(1e-12, abs(depth)))
    else:
        print("\nConsistency check skipped (NaNs).")

    # print whatever Tier4 said it computed, if summary is provided
    if args.tier4_summary:
        s = json.load(open(args.tier4_summary))
        print("\nTier4 summary computed flags:")
        print(json.dumps(s.get("computed", {}), indent=2))
        print("\nTier4 availability counts:")
        print(json.dumps(s.get("availability_counts", {}), indent=2))

if __name__ == "__main__":
    main()
