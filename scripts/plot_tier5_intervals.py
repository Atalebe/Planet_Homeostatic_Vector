#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier5", required=True, help="data/derived/tier5/tier5_scorecard.csv")
    ap.add_argument("--planet", required=True, help='planet_id, e.g. "LHS 1140 b"')
    ap.add_argument("--out", default="figures/tier5_intervals_lhs1140b.png")
    args = ap.parse_args()

    df = pd.read_csv(args.tier5, low_memory=False)
    df["planet_id"] = df["planet_id"].astype(str).str.strip()
    row = df.loc[df["planet_id"].eq(args.planet)]
    if len(row) == 0:
        raise SystemExit(f"Planet not found in Tier5 scorecard: {args.planet!r}")
    row = row.iloc[0]

    metrics = [
        ("Ripeness_T2", row["Ripeness_T2_med"], row["Ripeness_T2_p16"], row["Ripeness_T2_p84"]),
        ("Ripeness_T3", row["Ripeness_T3_med"], row["Ripeness_T3_p16"], row["Ripeness_T3_p84"]),
        ("Ripeness_T4", row["Ripeness_T4_med"], row["Ripeness_T4_p16"], row["Ripeness_T4_p84"]),
        ("Depth proxy", row["depth_proxy_med"], row["depth_proxy_p16"], row["depth_proxy_p84"]),
    ]

    # horizontal error bars
    y = list(range(len(metrics)))[::-1]
    labels = [m[0] for m in metrics][::-1]
    meds = [m[1] for m in metrics][::-1]
    p16s = [m[2] for m in metrics][::-1]
    p84s = [m[3] for m in metrics][::-1]
    xerr = [[med - p16 for med, p16 in zip(meds, p16s)],
            [p84 - med for med, p84 in zip(meds, p84s)]]

    plt.figure()
    plt.errorbar(meds, y, xerr=xerr, fmt="o", capsize=4)
    plt.yticks(y, labels)
    plt.xlabel("Value (median with 16th–84th percentile)")
    plt.title(f"Tier5 MC intervals: {args.planet} (n_mc={int(row['n_mc'])})")
    plt.tight_layout()
    plt.savefig(args.out, dpi=220)
    plt.close()
    print("Wrote:", args.out)

if __name__ == "__main__":
    main()
