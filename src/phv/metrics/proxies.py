from __future__ import annotations
import numpy as np
import pandas as pd

# Constants
AU = 1.0
RSUN_IN_AU = 0.00465047  # Rsun in AU (approx)
SIGMA_SB = 5.670374419e-8

def stellar_luminosity_Lsun(st_rad_rsun: pd.Series, st_teff_k: pd.Series) -> pd.Series:
    """
    Compute L/Lsun from radius and Teff using Stefan-Boltzmann scaling:
    L ~ R^2 T^4. Relative to Sun: (R/Rsun)^2 * (T/5772)^4
    """
    T_sun = 5772.0
    return (st_rad_rsun.astype(float)**2) * (st_teff_k.astype(float)/T_sun)**4

def log10_flux_rel_earth(L_Lsun: pd.Series, a_au: pd.Series) -> pd.Series:
    """
    Relative flux to Earth: F/F_earth ~ (L/Lsun) / a^2
    """
    return np.log10(np.maximum(L_Lsun.astype(float), 1e-12) / np.maximum(a_au.astype(float)**2, 1e-12))

def proxy_H(logF: pd.Series) -> pd.Series:
    # Heat throughput proxy
    return logF

def proxy_M(pl_bmasse: pd.Series | None, pl_rade: pd.Series | None) -> pd.Series:
    """
    Memory/retention proxy.
    Prefer log10(Mass) if available. If not, use log10(radius) as weak fallback.
    """
    if pl_bmasse is not None and pl_bmasse.notna().any():
        return np.log10(np.maximum(pl_bmasse.astype(float), 1e-12))
    if pl_rade is not None and pl_rade.notna().any():
        return np.log10(np.maximum(pl_rade.astype(float), 1e-12))
    # If neither exists, return NaNs
    return pd.Series(np.nan, index=(pl_bmasse.index if pl_bmasse is not None else pl_rade.index))

def proxy_S(pl_orbeccen: pd.Series | None,
            st_age: pd.Series | None) -> pd.Series:
    """
    Structural stability proxy:
    Penalize eccentricity. Optionally reward older stable systems via age (very light touch).
    """
    if pl_orbeccen is not None and pl_orbeccen.notna().any():
        e = pl_orbeccen.astype(float).fillna(np.nan)
        base = -np.abs(e)
    else:
        base = pd.Series(0.0, index=(st_age.index if st_age is not None else None))

    if st_age is not None and st_age.notna().any():
        # older systems, slightly higher S (proxy for long-term stability)
        age = st_age.astype(float)
        base = base + 0.1*np.log10(np.maximum(age, 0.1))
    return base

def proxy_R(logF: pd.Series, F_mid_log10: float, pl_bmasse: pd.Series | None) -> pd.Series:
    """
    Regeneration proxy:
    Favor moderate irradiation around midpoint and reward mass (internal longevity).
    """
    term_flux = -np.abs(logF.astype(float) - float(F_mid_log10))
    if pl_bmasse is not None and pl_bmasse.notna().any():
        term_mass = 0.2*np.log10(np.maximum(pl_bmasse.astype(float), 1e-12))
    else:
        term_mass = 0.0
    return term_flux + term_mass
