# Planet_Homeostatic_Vector

Computes a planetary homeostatic vector space (R,H,M,S) using exoplanet data,
then projects to Phi_p and applies a bounded window [Phi_min, Phi_max].

This does not detect life.
It operationalizes capacity for regulated stability and connects to a ripeness-style
residence-time proxy when stellar age exists.

## Input

Export a CSV from NASA Exoplanet Archive (or similar) and place it at:
data/raw/exoplanets/exoplanets.csv

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
# Planet_Homeostatic_Vector
