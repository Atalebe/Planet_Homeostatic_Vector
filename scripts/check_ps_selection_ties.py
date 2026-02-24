#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np

BASE_FIELDS = ["pl_rade","pl_bmasse","pl_orbeccen","pl_eqt","pl_insol","st_age"]

def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def completeness(row, fields):
    return int(np.sum([pd.notna(row.get(f)) for f in fields]))

def dist_to_tier4(row, baseline):
    # L2 norm over available baseline fields.
    d = 0.0
    n = 0
    for f, b in baseline.items():
        v = row.get(f)
        if pd.notna(v) and pd.notna(b):
            d += float(v - b) ** 2
            n += 1
    if n == 0:
        return np.inf
    return float(np.sqrt(d))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", required=True)
    ap.add_argument("--tier4", required=True)
    ap.add_argument("--planet", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t4 = pd.read_csv(args.tier4, low_memory=False)
    r = t4.loc[t4["planet_id"].astype(str).str.strip() == args.planet].iloc[0]
    baseline = {f: float(r[f]) for f in BASE_FIELDS}
    print("Tier4 baseline:", baseline)

    ps = pd.read_csv(args.ps, comment="#", low_memory=False)
    ps["pl_name"] = ps["pl_name"].astype(str).str.strip()
    sub = ps.loc[ps["pl_name"] == args.planet].copy()
    print("\nPS rows:", len(sub))
    if len(sub) == 0:
        raise SystemExit("No PS rows found for planet")

    for f in BASE_FIELDS:
        if f in sub.columns:
            sub[f] = to_num(sub[f])

    sub["dist_to_tier4"] = sub.apply(lambda row: dist_to_tier4(row, baseline), axis=1)
    sub["completeness"] = sub.apply(lambda row: completeness(row, BASE_FIELDS), axis=1)

    min_d = sub["dist_to_tier4"].min()
    tied = sub.loc[sub["dist_to_tier4"] == min_d].copy()
    tied = tied.sort_values(["completeness"], ascending=False)

    print("\nRows with minimum dist_to_tier4:")
    print(tied[["dist_to_tier4","completeness"] + BASE_FIELDS].to_string(index=False))

    sel = tied.iloc[0]
    print("\nRecommended robust selection:")
    print(" selected dist_to_tier4 =", float(sel["dist_to_tier4"]))
    print(" selected completeness  =", int(sel["completeness"]))
    print(" selected base fields:")
    for f in BASE_FIELDS:
        print(f"  {f:12s} = {sel.get(f)}")

    if args.out:
        sub.to_csv(args.out, index=False)
        print("\nWrote:", args.out)

if __name__ == "__main__":
    main()
