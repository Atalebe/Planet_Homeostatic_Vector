#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np

FIELDS = ["pl_rade","pl_bmasse","pl_orbeccen","pl_eqt","pl_insol","st_age"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", required=True)
    ap.add_argument("--planet", required=True)
    args = ap.parse_args()

    ps = pd.read_csv(args.ps, comment="#", low_memory=False)
    ps["pl_name"] = ps["pl_name"].astype(str).str.strip()
    sub = ps.loc[ps["pl_name"] == args.planet].copy()
    print("PS rows:", len(sub))
    for f in FIELDS:
        if f in sub.columns:
            sub[f] = pd.to_numeric(sub[f], errors="coerce")

    for f in FIELDS:
        if f in sub.columns:
            v = sub[f].dropna()
            if len(v) == 0:
                print(f"{f:12s}: all NaN")
            else:
                print(f"{f:12s}: n={len(v)}  min={v.min():g}  max={v.max():g}  std={v.std(ddof=1) if len(v)>1 else 0:g}")

    print("\nRaw rows (fields):")
    cols = ["pl_name"] + [c for c in FIELDS if c in sub.columns]
    print(sub[cols].head(20).to_string(index=False))

if __name__ == "__main__":
    main()
