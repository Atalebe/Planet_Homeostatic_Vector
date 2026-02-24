#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np

REARTH_PER_RSUN = 109.076  # approx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier4", required=True)
    ap.add_argument("--planet", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.tier4, low_memory=False)
    row = df.loc[df["planet_id"].astype(str).str.strip() == args.planet].iloc[0]

    rp = float(row["pl_rade"])                 # Earth radii
    rs = float(row["st_rad"]) * REARTH_PER_RSUN # convert Rsun -> Rearth
    stored = float(row["tran_depth_proxy"]) if "tran_depth_proxy" in df.columns else np.nan
    recomputed = (rp/rs)**2

    print("Inputs from tier4_enriched:")
    print(" planet_id       =", args.planet)
    print(" pl_rade [Re]    =", rp)
    print(" st_rad  [Rsun]  =", float(row["st_rad"]))
    print(" st_rad  [Re]    =", rs)
    print(" stored tran_depth_proxy =", stored)

    print("\nRecomputed proxy:")
    print(" depth_proxy = (Rp/Rs)^2 =", recomputed)

    if np.isfinite(stored):
        ad = abs(recomputed - stored)
        rd = ad / stored if stored != 0 else np.nan
        print("\nConsistency check:")
        print(" abs diff =", ad)
        print(" rel diff =", rd)

if __name__ == "__main__":
    main()
