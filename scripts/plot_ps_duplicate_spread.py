#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIELDS = [
    ("pl_rade", "Planet radius $R_p$ [$R_\\oplus$]"),
    ("pl_bmasse", "Planet mass $M_p$ [$M_\\oplus$]"),
    ("pl_orbeccen", "Eccentricity $e$"),
    ("pl_eqt", "Equilibrium temperature $T_{eq}$ [K]"),
    ("pl_insol", "Insolation $S$ [$S_\\oplus$]"),
    ("st_age", "Stellar age [Gyr]"),
]

def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", required=True, help="PS_*.csv from NASA Exoplanet Archive")
    ap.add_argument("--diag", required=True, help="tier5 ps_selection_diagnostic_*.csv")
    ap.add_argument("--planet", required=True, help='planet name, e.g. "LHS 1140 b"')
    ap.add_argument("--outdir", default="figures", help="output directory")
    args = ap.parse_args()

    ps = pd.read_csv(args.ps, comment="#", low_memory=False)
    ps["pl_name"] = ps["pl_name"].astype(str).str.strip()

    sub = ps.loc[ps["pl_name"].eq(args.planet)].copy()
    if len(sub) == 0:
        raise SystemExit(f"No PS rows found for planet={args.planet!r}")

    # Diagnostic file includes dist_to_tier4 and completeness; use it to mark selected row.
    diag = pd.read_csv(args.diag, low_memory=False)
    # Keep only rows for this planet
    diag = diag.loc[diag["pl_name"].astype(str).str.strip().eq(args.planet)].copy()
    if len(diag) == 0:
        raise SystemExit(f"No diagnostic rows found in {args.diag} for planet={args.planet!r}")

    # Determine "selected" row in diagnostic: minimum dist_to_tier4, then max completeness, then first.
    diag["dist_to_tier4"] = numeric(diag.get("dist_to_tier4"))
    diag["completeness"] = numeric(diag.get("completeness"))
    diag_sorted = diag.sort_values(["dist_to_tier4", "completeness"], ascending=[True, False])
    selected = diag_sorted.iloc[0]

    # We don’t have stable IDs for PS duplicates, so we mark selected by matching the baseline fields.
    # This works because the selected row matches Tier4 baseline exactly for the 6 baseline fields.
    sel_mask = np.ones(len(sub), dtype=bool)
    for col, _ in FIELDS:
        if col not in sub.columns:
            continue
        v = selected.get(col, np.nan)
        s = numeric(sub[col])
        if pd.isna(v):
            # selected baseline never NaN for these, but keep safe
            sel_mask &= s.isna().to_numpy()
        else:
            sel_mask &= np.isclose(s.to_numpy(), float(v), rtol=0, atol=1e-12) | (s.isna().to_numpy() & False)

    # If multiple match (rare), keep them all highlighted.
    is_selected = sel_mask

    import os
    os.makedirs(args.outdir, exist_ok=True)

    for col, label in FIELDS:
        if col not in sub.columns:
            continue
        y = numeric(sub[col]).to_numpy()
        x = np.arange(len(y))

        plt.figure()
        plt.scatter(x[~np.isnan(y)], y[~np.isnan(y)], marker="o", label="PS duplicates")
        if is_selected.any():
            ys = y[is_selected]
            xs = x[is_selected]
            plt.scatter(xs[~np.isnan(ys)], ys[~np.isnan(ys)], marker="*", s=140, label="Selected PS row")

        plt.xlabel("PS duplicate row index")
        plt.ylabel(label)
        plt.title(f"{args.planet}: PS duplicate spread ({col})")
        plt.legend()
        plt.tight_layout()
        out = f"{args.outdir}/ps_duplicate_spread_{args.planet.replace(' ','_')}_{col}.png"
        plt.savefig(out, dpi=200)
        plt.close()

    print(f"Wrote plots to: {args.outdir}/ps_duplicate_spread_*")

if __name__ == "__main__":
    main()
