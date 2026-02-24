#!/usr/bin/env python3
import argparse
import json
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier5summary", required=True)
    ap.add_argument("--tier5score", required=True)
    args = ap.parse_args()

    s = json.load(open(args.tier5summary))
    print("summary keys:", sorted(s.keys()))
    print("ps_merge present:", "ps_merge" in s)
    if "ps_merge" in s:
        print("ps_merge:", s["ps_merge"])
    print("use_ps_errors:", s.get("use_ps_errors"))
    print("use_floors_when_missing:", s.get("use_floors_when_missing"))
    print("floors:", s.get("floors"))

    df = pd.read_csv(args.tier5score, low_memory=False)
    row = df.iloc[0]

    pairs = [
        ("Ripeness_T2", "Ripeness_T2_med", "Ripeness_T2_p16", "Ripeness_T2_p84"),
        ("Ripeness_T3", "Ripeness_T3_med", "Ripeness_T3_p16", "Ripeness_T3_p84"),
        ("Ripeness_T4", "Ripeness_T4_med", "Ripeness_T4_p16", "Ripeness_T4_p84"),
        ("depth_proxy", "depth_proxy_med", "depth_proxy_p16", "depth_proxy_p84"),
    ]
    print("\nDegeneracy check (med==p16==p84 means broken MC):")
    for name, m, p16, p84 in pairs:
        a, b, c = row[m], row[p16], row[p84]
        print(f"{name:10s}: med={a}  p16={b}  p84={c}  degenerate={a==b==c}")

if __name__ == "__main__":
    main()
