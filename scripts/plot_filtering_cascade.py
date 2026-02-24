#!/usr/bin/env python3
"""
Plot the PHV filtering cascade as a simple horizontal bar chart.

Default behavior:
- Tries to infer counts from tier summary.json files inside data/derived/tier*/summary.json
- If some tiers are missing, it still plots what it can.
- Optional explicit overrides via CLI.

Output:
- figures/phv_filtering_cascade.png (default)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


@dataclass
class Stage:
    label: str
    count: int


def _read_json(path: str) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_counts_from_summaries(base: str) -> Dict[str, int]:
    """
    Attempts to infer the key cascade counts from available summary.json files.
    This is intentionally conservative: only uses fields that are likely to exist.
    """
    out: Dict[str, int] = {}

    # Tier summaries if present
    t2 = _read_json(os.path.join(base, "tier2", "summary.json"))
    t3 = _read_json(os.path.join(base, "tier3", "summary.json"))
    t4 = _read_json(os.path.join(base, "tier4", "summary.json"))
    t5 = _read_json(os.path.join(base, "tier5", "summary.json"))

    # Tier4/5 have reliable n_input/n_output
    if t4:
        out["tier4_n_in"] = int(t4.get("n_input", 0) or 0)
        out["tier4_n_out"] = int(t4.get("n_output", 0) or 0)
    if t5:
        out["tier5_n_in"] = int(t5.get("n_input", 0) or 0)
        out["tier5_n_out"] = int(t5.get("n_output", 0) or 0)

    # Tier2/3 vary by implementation; try the same keys
    if t2:
        out["tier2_n_in"] = int(t2.get("n_input", 0) or 0)
        out["tier2_n_out"] = int(t2.get("n_output", 0) or 0)
    if t3:
        out["tier3_n_in"] = int(t3.get("n_input", 0) or 0)
        out["tier3_n_out"] = int(t3.get("n_output", 0) or 0)

    return out


def build_stages(
    raw_rows: int,
    required_rows: int,
    unique_default: int,
    rocky_total: int,
    rocky_phi_finite: int,
    rocky_in_window: int,
    tier2_survivors: int,
    tier3_survivors: int,
    tier4_survivors: int,
) -> List[Stage]:
    stages = [
        Stage("Raw archive rows (PS export)", raw_rows),
        Stage("After required columns (P, a, Teff, R*)", required_rows),
        Stage("default_flag=1 + dedup (unique planets)", unique_default),
        Stage("Rocky subset (Rp ≤ 1.8 R⊕)", rocky_total),
        Stage("Rocky with finite Φp", rocky_phi_finite),
        Stage("Rocky in class window (40–60%)", rocky_in_window),
        Stage("Tier 2 hard gates + age required", tier2_survivors),
        Stage("Tier 3 derived physics checks", tier3_survivors),
        Stage("Tier 4 follow-up layer", tier4_survivors),
    ]
    # Keep only valid positives
    return [s for s in stages if isinstance(s.count, int) and s.count > 0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived", default="data/derived", help="Base folder containing tier*/summary.json")
    ap.add_argument("--out", default="figures/phv_filtering_cascade.png", help="Output PNG path")

    # Baseline cascade counts (defaults match what you already reported in the paper table)
    ap.add_argument("--raw-rows", type=int, default=39386)
    ap.add_argument("--required-rows", type=int, default=19854)
    ap.add_argument("--unique-default", type=int, default=3012)
    ap.add_argument("--rocky-total", type=int, default=600)
    ap.add_argument("--rocky-phi-finite", type=int, default=40)
    ap.add_argument("--rocky-in-window", type=int, default=8)

    # Tier survivors can be inferred from summary files if present
    ap.add_argument("--tier2-survivors", type=int, default=-1)
    ap.add_argument("--tier3-survivors", type=int, default=-1)
    ap.add_argument("--tier4-survivors", type=int, default=-1)

    args = ap.parse_args()

    inferred = _infer_counts_from_summaries(args.derived)

    # Prefer explicit CLI if provided, else infer, else fallback to 1 for strict run
    tier2 = args.tier2_survivors if args.tier2_survivors >= 0 else inferred.get("tier2_n_out", 1)
    tier3 = args.tier3_survivors if args.tier3_survivors >= 0 else inferred.get("tier3_n_out", 1)
    tier4 = args.tier4_survivors if args.tier4_survivors >= 0 else inferred.get("tier4_n_out", 1)

    stages = build_stages(
        raw_rows=args.raw_rows,
        required_rows=args.required_rows,
        unique_default=args.unique_default,
        rocky_total=args.rocky_total,
        rocky_phi_finite=args.rocky_phi_finite,
        rocky_in_window=args.rocky_in_window,
        tier2_survivors=int(tier2),
        tier3_survivors=int(tier3),
        tier4_survivors=int(tier4),
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Plot: horizontal bars, log scale helps show collapse clearly
    labels = [s.label for s in stages][::-1]
    counts = [s.count for s in stages][::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(counts)), counts)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Count")
    plt.xscale("log")
    plt.title("PHV filtering cascade (strict rocky configuration)")
    for i, v in enumerate(counts):
        plt.text(v, i, f"  {v}", va="center")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"[OK] wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
