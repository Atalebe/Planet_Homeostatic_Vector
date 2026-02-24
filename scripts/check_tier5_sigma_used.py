# scripts/check_tier5_sigma_used.py
# Usage:
#   .venv/bin/python scripts/check_tier5_sigma_used.py \
#     --ps data/raw/exoplanets/PS_2026.02.23_02.08.36.csv \
#     --tier4 data/derived/tier4/tier4_enriched.csv \
#     --tier5summary data/derived/tier5/summary.json \
#     --planet "LHS 1140 b"
#
# What it does:
# - loads floors from tier5 summary.json
# - loads PS duplicates and selects row "closest_to_tier4" (same metric as earlier)
#!/usr/bin/env python3
import argparse
import json
import numpy as np
import pandas as pd

PARAMS = [
    ("pl_rade", "pl_radeerr1", "pl_radeerr2", "floor_frac_rade", "frac"),
    ("pl_bmasse", "pl_bmasseerr1", "pl_bmasseerr2", "floor_frac_bmasse", "frac"),
    ("pl_orbeccen", "pl_orbeccenerr1", "pl_orbeccenerr2", "floor_abs_ecc", "abs"),
    ("pl_eqt", "pl_eqterr1", "pl_eqterr2", "floor_frac_teq", "frac"),
    ("pl_insol", "pl_insolerr1", "pl_insolerr2", "floor_frac_insol", "frac"),
    ("st_age", "st_ageerr1", "st_ageerr2", "floor_abs_age_gyr", "abs"),
]

BASE_FIELDS = ["pl_rade","pl_bmasse","pl_orbeccen","pl_eqt","pl_insol","st_age"]

def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def sigma_from_err(err1, err2):
    # PS convention: err1 is +, err2 is -
    # Use symmetric approx = mean(|err1|, |err2|)
    a = np.nan
    if pd.notna(err1) or pd.notna(err2):
        vals = []
        if pd.notna(err1): vals.append(abs(float(err1)))
        if pd.notna(err2): vals.append(abs(float(err2)))
        if vals:
            a = float(np.mean(vals))
    return a

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", required=True)
    ap.add_argument("--tier4", required=True)
    ap.add_argument("--tier5summary", required=True)
    ap.add_argument("--planet", required=True)
    args = ap.parse_args()

    s = json.load(open(args.tier5summary))
    floors = s["floors"]

    t4 = pd.read_csv(args.tier4, low_memory=False)
    r = t4.loc[t4["planet_id"].astype(str).str.strip() == args.planet].iloc[0]
    baseline = {f: float(r[f]) for f in BASE_FIELDS}

    ps = pd.read_csv(args.ps, comment="#", low_memory=False)
    ps["pl_name"] = ps["pl_name"].astype(str).str.strip()
    sub = ps.loc[ps["pl_name"] == args.planet].copy()

    # numeric conversion
    for c in set([p[0] for p in PARAMS] + [p[1] for p in PARAMS] + [p[2] for p in PARAMS]):
        if c in sub.columns:
            sub[c] = to_num(sub[c])

    # compute dist to tier4 to mimic selection
    def dist(row):
        d=0.0; n=0
        for f,b in baseline.items():
            v=row.get(f)
            if pd.notna(v) and pd.notna(b):
                d += float(v-b)**2; n+=1
        return np.inf if n==0 else float(np.sqrt(d))

    sub["dist_to_tier4"] = sub.apply(dist, axis=1)
    sub["completeness"] = sub.apply(lambda row: int(np.sum([pd.notna(row.get(f)) for f in BASE_FIELDS])), axis=1)

    min_d = sub["dist_to_tier4"].min()
    tied = sub.loc[sub["dist_to_tier4"] == min_d].sort_values("completeness", ascending=False)
    sel = tied.iloc[0]

    print("\nSelected PS row:")
    print(" dist_to_tier4 =", float(sel["dist_to_tier4"]))
    print(" completeness  =", int(sel["completeness"]))

    print("\nSigma used per parameter (PS vs floors):")
    print(" floors =", floors)

    for base, e1, e2, floor_key, floor_kind in PARAMS:
        basev = baseline[base]
        ps_sig = sigma_from_err(sel.get(e1, np.nan), sel.get(e2, np.nan))
        if floor_kind == "frac":
            floor_sig = abs(basev) * float(floors[floor_key])
        else:
            floor_sig = float(floors[floor_key])

        if pd.isna(ps_sig) or ps_sig == 0:
            used = floor_sig
            src = "FLOOR"
        else:
            used = ps_sig
            src = "PS"

        print(f" {base:11s} base={basev:10g}  sigma_ps={ps_sig:10g}  sigma_floor={floor_sig:10g}  sigma_used={used:10g}  src={src}")

if __name__ == "__main__":
    main()
