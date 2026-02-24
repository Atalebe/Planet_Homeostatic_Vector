# src/phv/cli.py
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Optional

import yaml

from phv.pipeline.run import run_pipeline
from phv.pipeline.tier2 import run_tier2
from phv.pipeline.tier3 import run_tier3
from phv.pipeline.tier4 import run_tier4
from phv.pipeline.tier5 import run_tier5


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping (dict). Got: {type(data)}")
    return data


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phv", description="Planet Homeostatic Vector (PHV) pipeline CLI")
    subparsers = p.add_subparsers(dest="cmd", required=True)

    # Tier1: "run"
    sp_run = subparsers.add_parser("run", help="Tier 1: compute Phi_p and class-relative window membership")
    sp_run.add_argument("--config", required=True, help="Path to YAML config")

    # Tier2
    sp_t2 = subparsers.add_parser("tier2", help="Tier 2: apply strict thermodynamic + age gates and rank candidates")
    sp_t2.add_argument("--config", required=True, help="Path to YAML config")

    # Tier3
    sp_t3 = subparsers.add_parser("tier3", help="Tier 3: derive density, surface gravity, escape-speed and scorecard")
    sp_t3.add_argument("--config", required=True, help="Path to YAML config")

    # Tier4
    sp_t4 = subparsers.add_parser("tier4", help="Tier 4: merge PS photometry/distances and follow-up score")
    sp_t4.add_argument("--config", required=True, help="Path to YAML config")

    # Tier5
    sp_t5 = subparsers.add_parser("tier5", help="Tier 5: Monte Carlo uncertainty propagation + probabilistic scorecard")
    sp_t5.add_argument("--config", required=True, help="Path to YAML config")

    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)

    if args.cmd == "run":
        run_pipeline(cfg, config_path=args.config)
    elif args.cmd == "tier2":
        run_tier2(cfg, config_path=args.config)
    elif args.cmd == "tier3":
        run_tier3(cfg, config_path=args.config)
    elif args.cmd == "tier4":
        run_tier4(cfg, config_path=args.config)
    elif args.cmd == "tier5":
        run_tier5(cfg, config_path=args.config)
    else:
        # should never happen because argparse enforces choices
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
