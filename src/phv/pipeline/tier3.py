from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd


@dataclass
class Tier3Config:
    run_id: str
    out_dir: str
    tier2_candidates_csv: str
    rho_earth_g_cm3: float = 5.514
    g_earth_m_s2: float = 9.80665
    require_tier2_pass: bool = True
    strict_derived: bool = True
    followup_factor_default: float = 1.0


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def compute_derived(df: pd.DataFrame, cfg: Tier3Config) -> pd.DataFrame:
    """
    Add basic derived metrics in Earth units where possible.
    Uses:
      - pl_rade [R_earth]
      - pl_bmasse [M_earth]
    Produces:
      - rho_bulk_g_cm3
      - g_surface_m_s2
      - vesc_rel_earth  (escape velocity relative to Earth)
    """
    out = df.copy()

    if "pl_rade" in out.columns:
        out["pl_rade"] = _num(out["pl_rade"])
    if "pl_bmasse" in out.columns:
        out["pl_bmasse"] = _num(out["pl_bmasse"])

    r = out["pl_rade"] if "pl_rade" in out.columns else pd.Series(np.nan, index=out.index)
    m = out["pl_bmasse"] if "pl_bmasse" in out.columns else pd.Series(np.nan, index=out.index)

    ok = np.isfinite(r) & np.isfinite(m) & (r > 0) & (m > 0)

    # density in Earth units: rho/rho_earth = M / R^3
    rho_rel = pd.Series(np.nan, index=out.index)
    rho_rel.loc[ok] = (m.loc[ok] / (r.loc[ok] ** 3))

    # Convert to g/cm^3 using Earth density constant
    out["rho_bulk_g_cm3"] = rho_rel * cfg.rho_earth_g_cm3

    # surface gravity relative to Earth: g/g_earth = M / R^2
    g_rel = pd.Series(np.nan, index=out.index)
    g_rel.loc[ok] = (m.loc[ok] / (r.loc[ok] ** 2))
    out["g_surface_m_s2"] = g_rel * cfg.g_earth_m_s2

    # escape velocity relative to Earth: vesc/vesc_earth = sqrt(M / R)
    vesc_rel = pd.Series(np.nan, index=out.index)
    vesc_rel.loc[ok] = np.sqrt(m.loc[ok] / r.loc[ok])
    out["vesc_rel_earth"] = vesc_rel

    if cfg.strict_derived:
        # keep derived NaN where inputs missing; no imputation
        pass

    return out


def score_tier3(df: pd.DataFrame, cfg: Tier3Config) -> pd.DataFrame:
    """
    Minimal scoring: Ripeness_T3 = Ripeness_T2 * followup_factor.
    followup_factor is currently constant; later you can upgrade it using:
      - sy_vmag / sy_gaiamag / sy_dist (if available)
      - transit depth, star brightness, etc.
    """
    out = df.copy()

    if "Ripeness_T2" in out.columns:
        out["Ripeness_T2"] = _num(out["Ripeness_T2"])
    else:
        out["Ripeness_T2"] = np.nan

    out["followup_factor"] = float(cfg.followup_factor_default)
    out["Ripeness_T3"] = out["Ripeness_T2"] * out["followup_factor"]

    # rank highest first, NaNs last
    out.sort_values(["Ripeness_T3"], ascending=False, inplace=True, na_position="last")
    out["rank_tier3"] = np.arange(1, len(out) + 1)

    return out


def run_tier3(cfg_dict: Dict[str, Any], config_path: str | None = None) -> None:
    run_id = cfg_dict["run"]["run_id"]
    out_dir = cfg_dict["run"]["out_dir"]

    cfg = Tier3Config(
        run_id=run_id,
        out_dir=out_dir,
        tier2_candidates_csv=cfg_dict["inputs"]["tier2_candidates_csv"],
        rho_earth_g_cm3=float(cfg_dict.get("constants", {}).get("rho_earth_g_cm3", 5.514)),
        g_earth_m_s2=float(cfg_dict.get("constants", {}).get("g_earth_m_s2", 9.80665)),
        require_tier2_pass=bool(cfg_dict.get("tier3", {}).get("require_tier2_pass", True)),
        strict_derived=bool(cfg_dict.get("tier3", {}).get("strict_derived", True)),
        followup_factor_default=float(cfg_dict.get("tier3", {}).get("followup_factor_default", 1.0)),
    )

    _ensure_dir(cfg.out_dir)

    df = pd.read_csv(cfg.tier2_candidates_csv)

    if cfg.require_tier2_pass and "tier2_pass" in df.columns:
        df["tier2_pass"] = _num(df["tier2_pass"]).fillna(0).astype(int)
        df = df[df["tier2_pass"] == 1].copy()

    n_in = int(len(df))

    df = compute_derived(df, cfg)
    df = score_tier3(df, cfg)

    out_enriched = os.path.join(cfg.out_dir, "tier3_enriched.csv")
    out_scorecard = os.path.join(cfg.out_dir, "tier3_scorecard.csv")
    out_summary = os.path.join(cfg.out_dir, "summary.json")
    out_meta = os.path.join(cfg.out_dir, "run_meta.json")

    df.to_csv(out_enriched, index=False)

    # scorecard: a tight column subset for manuscript use
    cols = [
        "rank_tier3", "planet_id", "hostname",
        "pl_rade", "pl_bmasse", "st_age",
        "pl_insol", "pl_eqt", "pl_orbeccen",
        "Phi_p", "Ripeness_T2", "rho_bulk_g_cm3", "g_surface_m_s2", "vesc_rel_earth",
        "Ripeness_T3",
    ]
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(out_scorecard, index=False)

    summary = {
        "run_id": cfg.run_id,
        "tier2_candidates_csv": cfg.tier2_candidates_csv,
        "n_input_after_optional_tier2_pass_filter": n_in,
        "n_output": int(len(df)),
        "derived_available_counts": {
            "rho_bulk_g_cm3": int(pd.to_numeric(df.get("rho_bulk_g_cm3", np.nan), errors="coerce").notna().sum()),
            "g_surface_m_s2": int(pd.to_numeric(df.get("g_surface_m_s2", np.nan), errors="coerce").notna().sum()),
            "vesc_rel_earth": int(pd.to_numeric(df.get("vesc_rel_earth", np.nan), errors="coerce").notna().sum()),
        },
        "outputs": {
            "tier3_enriched_csv": out_enriched,
            "tier3_scorecard_csv": out_scorecard,
        },
    }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    meta = {
        "run_id": cfg.run_id,
        "config_path": config_path,
        "inputs": {"tier2_candidates_csv": cfg.tier2_candidates_csv},
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[TIER3] Run: {cfg.run_id}")
    print(f"[TIER3] Wrote: {out_enriched}")
    print(f"[TIER3] Wrote: {out_scorecard}")
    print(f"[TIER3] Wrote: {out_summary}")
    print(f"[TIER3] Wrote: {out_meta}")
    print("[TIER3] Done.")
