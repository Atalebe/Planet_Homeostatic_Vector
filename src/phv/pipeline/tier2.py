# src/phv/pipeline/tier2.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _classify_radius(pl_rade: float, rocky_rmax: float, subnep_rmax: float) -> str:
    if not np.isfinite(pl_rade) or pl_rade <= 0:
        return "unknown"
    if pl_rade <= rocky_rmax:
        return "rocky"
    if pl_rade <= subnep_rmax:
        return "subnep"
    return "giant"


def _phi_closeness(phi: float, phi_min: float, phi_max: float) -> float:
    if not np.isfinite(phi) or not np.isfinite(phi_min) or not np.isfinite(phi_max):
        return np.nan
    c = 0.5 * (phi_min + phi_max)
    hw = 0.5 * (phi_max - phi_min)
    if hw <= 0:
        return np.nan
    z = (phi - c) / hw
    return float(np.exp(-(z ** 2)))


def _age_factor(age_gyr: float, age_scale: float) -> float:
    if not np.isfinite(age_gyr) or age_gyr <= 0:
        return np.nan
    return float(min(1.0, age_gyr / max(1e-6, age_scale)))


def _num(x):
    return pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]

def _hard_gate(row: pd.Series, cfg: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    d: Dict[str, Any] = {}

    require_age = bool(cfg["tier2"].get("require_age", True))
    require_rocky = bool(cfg["tier2"].get("require_rocky", False))  # NEW
    if cfg["tier2"].get("require_in_window", False):
        if int(row.get("in_window", 0)) != 1:
            return False, {"fail": "not_in_window"}
    # numeric coercion (prevents string weirdness)
    age  = _num(row.get("st_age", np.nan))
    insol = _num(row.get("pl_insol", np.nan))
    teq  = _num(row.get("pl_eqt", np.nan))
    ecc  = _num(row.get("pl_orbeccen", np.nan))
    rade = _num(row.get("pl_rade", np.nan))

    if require_age and not np.isfinite(age):
        d["fail"] = "missing_age"; return False, d

    # if we care about life plausibility, enforce rocky
    if require_rocky:
        rocky_rmax = float(cfg["tier2"].get("rocky_rmax_rearth", 1.8))
    if not np.isfinite(rade):
            d["fail"] = "missing_radius"; return False, d
    if rade <= 0 or rade > rocky_rmax:
            d["fail"] = "not_rocky"; return False, d

    # hard requires known insol, teq, ecc
    if not np.isfinite(insol):
        d["fail"] = "missing_insol"; return False, d
    if not np.isfinite(teq):
        d["fail"] = "missing_teq"; return False, d
    if not np.isfinite(ecc):
        d["fail"] = "missing_ecc"; return False, d

    Smin, Smax = float(cfg["tier2"]["S_min"]), float(cfg["tier2"]["S_max"])
    Tmin, Tmax = float(cfg["tier2"]["Teq_min_K"]), float(cfg["tier2"]["Teq_max_K"])
    emax = float(cfg["tier2"]["e_max"])

    if insol < Smin or insol > Smax:
        d["fail"] = "insol_outside"; return False, d
    if teq < Tmin or teq > Tmax:
        d["fail"] = "teq_outside"; return False, d
    if ecc > emax:
        d["fail"] = "ecc_outside"; return False, d

    return True, d

def _soft_h_multiplier(row: pd.Series, cfg: Dict[str, Any]) -> float:
    """
    Soft habitability multiplier in [0,1].
    - Missing key fields => multiply by missing_penalty.
    - Deviations from S and Teq windows => Gaussian penalty.
    - Eccentricity penalty if known, else missing penalty.
    """
    miss_pen = float(cfg["tier2"].get("missing_penalty", 0.25))

    insol = row.get("pl_insol", np.nan)
    teq = row.get("pl_eqt", np.nan)
    ecc = row.get("pl_orbeccen", np.nan)

    Smin, Smax = float(cfg["tier2"]["S_min"]), float(cfg["tier2"]["S_max"])
    Tmin, Tmax = float(cfg["tier2"]["Teq_min_K"]), float(cfg["tier2"]["Teq_max_K"])
    emax = float(cfg["tier2"]["e_max"])

    # Insolation penalty in log-space
    if np.isfinite(insol) and insol > 0:
        Smid = np.sqrt(Smin * Smax)
        sigmaS = float(cfg["tier2"].get("soft_sigma_S", 0.7))
        zS = (np.log(insol) - np.log(Smid)) / max(1e-6, sigmaS)
        pS = float(np.exp(-(zS ** 2)))
    else:
        pS = miss_pen

    # Teq penalty
    if np.isfinite(teq):
        Tmid = 0.5 * (Tmin + Tmax)
        sigmaT = float(cfg["tier2"].get("soft_sigma_T", 60.0))
        zT = (teq - Tmid) / max(1e-6, sigmaT)
        pT = float(np.exp(-(zT ** 2)))
    else:
        pT = miss_pen

    # Ecc penalty: clamp if known, else miss penalty
    if np.isfinite(ecc):
        if ecc <= emax:
            pE = 1.0
        else:
            # linear falloff beyond emax
            pE = float(max(0.0, 1.0 - (ecc - emax) / max(1e-6, emax)))
    else:
        pE = miss_pen

    return float(np.clip(pS * pT * pE, 0.0, 1.0))


def run_tier2(cfg: Dict[str, Any], config_path: str) -> None:
    out_dir = cfg["run"]["out_dir"]
    _ensure_dir(out_dir)

    tier1_table = cfg["inputs"]["tier1_table"]
    tier1_summary = cfg["inputs"]["tier1_summary"]

    df = pd.read_csv(tier1_table)
    summ = _load_json(tier1_summary)

    rocky_rmax = float(cfg["tier2"].get("rocky_rmax_rearth", 1.8))
    subnep_rmax = float(cfg["tier2"].get("subnep_rmax_rearth", 4.0))

    # Attach class label
    df["pclass"] = [
        _classify_radius(x, rocky_rmax, subnep_rmax) for x in pd.to_numeric(df["pl_rade"], errors="coerce").to_numpy()
    ]

    # Grab per-class window bounds from tier1 summary (quantile_by_class)
    wb = summ.get("window_bounds", {})
    def get_bounds(cls: str) -> Tuple[float, float]:
        if cls in wb and "Phi_min" in wb[cls] and "Phi_max" in wb[cls]:
            return float(wb[cls]["Phi_min"]), float(wb[cls]["Phi_max"])
        # fallback: global bounds if present
        if "global" in wb:
            return float(wb["global"]["Phi_min"]), float(wb["global"]["Phi_max"])
        return (np.nan, np.nan)

    phi_min = []
    phi_max = []
    for cls in df["pclass"].tolist():
        pmin, pmax = get_bounds(cls)
        phi_min.append(pmin)
        phi_max.append(pmax)
    df["Phi_min_cls"] = phi_min
    df["Phi_max_cls"] = phi_max

    # Closeness + age factor
    phi = pd.to_numeric(df["Phi_p"], errors="coerce").to_numpy()
    pmin = pd.to_numeric(df["Phi_min_cls"], errors="coerce").to_numpy()
    pmax = pd.to_numeric(df["Phi_max_cls"], errors="coerce").to_numpy()

    clos = np.array([_phi_closeness(a, b, c) for a, b, c in zip(phi, pmin, pmax)], dtype=float)
    df["C_phi"] = clos

    age_scale = float(cfg["tier2"].get("age_scale_Gyr", 4.5))
    age = pd.to_numeric(df.get("st_age", np.nan), errors="coerce").to_numpy()
    Af = np.array([_age_factor(a, age_scale) for a in age], dtype=float)
    df["A_age"] = Af

    # Habitability multiplier and gate
    mode = str(cfg["tier2"].get("mode", "hard")).strip().lower()
    if mode not in ("hard", "soft"):
        raise ValueError(f"tier2.mode must be hard or soft, got: {mode}")

    hard_pass = []
    hard_fail = []
    h_mult = []

    for _, row in df.iterrows():
        if mode == "hard":
            ok, diag = _hard_gate(row, cfg)
            hard_pass.append(int(ok))
            hard_fail.append(diag.get("fail", ""))
            h_mult.append(1.0 if ok else 0.0)
        else:
            ok, diag = _hard_gate(row, cfg)  # still compute fail reason, but don't drop; just downweight
            hard_pass.append(int(ok))
            hard_fail.append(diag.get("fail", ""))
            h_mult.append(_soft_h_multiplier(row, cfg))

    df["tier2_pass"] = hard_pass
    df["tier2_fail_reason"] = hard_fail
    df["H_mult"] = np.array(h_mult, dtype=float)

    # Ripeness score
    power = float(cfg["tier2"].get("phi_closeness_power", 1.0))
    df["C_phi_pow"] = np.power(df["C_phi"], power)

    df["Ripeness_T2"] = df["C_phi_pow"] * df["A_age"] * df["H_mult"]

    # Candidate table
    # In hard mode keep only pass; in soft mode keep all but rank by Ripeness_T2
    if mode == "hard":
        cand = df[df["tier2_pass"] == 1].copy()
    else:
        cand = df.copy()

    cand.sort_values(["Ripeness_T2", "C_phi", "A_age"], ascending=[False, False, False], inplace=True)

    # Write outputs
    cand_csv = os.path.join(out_dir, "tier2_candidates_ranked.csv")
    cand.to_csv(cand_csv, index=False)

    # Top-k
    top_k = int(cfg.get("outputs", {}).get("top_k", 100))
    top_csv = os.path.join(out_dir, f"tier2_top_{top_k}.csv")
    cand.head(top_k).to_csv(top_csv, index=False)

    # Anchor report
    anchor_path = cfg["inputs"].get("rocky_anchor_csv", "")
    anchor_report = {}
    if anchor_path and os.path.exists(anchor_path):
        a = pd.read_csv(anchor_path)
        anchor_ids = set(a["planet_id"].astype(str).tolist())
        sub = cand[cand["planet_id"].astype(str).isin(anchor_ids)].copy()
        sub.sort_values(["Ripeness_T2"], ascending=[False], inplace=True)
        out_a = os.path.join(out_dir, "tier2_rocky_anchor_report.csv")
        sub.to_csv(out_a, index=False)
        anchor_report = {
            "anchor_n": int(len(anchor_ids)),
            "anchor_found_in_candidates": int(len(sub)),
            "anchor_report_csv": out_a,
        }

    # Summary JSON
    summ_out = {
        "run_id": cfg["run"]["run_id"],
        "mode": mode,
        "tier1_table": tier1_table,
        "tier1_summary": tier1_summary,
        "n_input": int(len(df)),
        "n_candidates": int(len(cand)),
        "top_k": top_k,
        "outputs": {
            "tier2_candidates_ranked_csv": cand_csv,
            "tier2_top_k_csv": top_csv,
        },
        "anchor_report": anchor_report,
        "gates": {
            "S_min": float(cfg["tier2"]["S_min"]),
            "S_max": float(cfg["tier2"]["S_max"]),
            "Teq_min_K": float(cfg["tier2"]["Teq_min_K"]),
            "Teq_max_K": float(cfg["tier2"]["Teq_max_K"]),
            "e_max": float(cfg["tier2"]["e_max"]),
            "require_age": bool(cfg["tier2"].get("require_age", True)),
            "require_rocky": bool(cfg["tier2"].get("require_rocky", False)),
            "rocky_rmax_rearth": float(cfg["tier2"].get("rocky_rmax_rearth", 1.8)),
        },
        "age_scale_Gyr": age_scale,
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ_out, f, indent=2)

    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        meta = {
            "run_id": cfg["run"]["run_id"],
            "config_path": config_path,
            "input_tier1_table": tier1_table,
            "input_tier1_summary": tier1_summary,
        }
        json.dump(meta, f, indent=2)
