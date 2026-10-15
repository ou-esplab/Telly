"""
Shared config-loading/path helpers for the ATM workflow scripts.

load_config() merges config/defaults.yaml (repo-wide defaults shared by
every experiment) underneath the experiment's own YAML file, which always
wins on any key it sets explicitly.
"""

import os

import yaml


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    defaults_path = os.path.join(repo_root, "config", "defaults.yaml")
    if os.path.exists(defaults_path):
        with open(defaults_path) as f:
            defaults = yaml.safe_load(f) or {}
        merged = {**defaults, **cfg}
        return merged

    return cfg


SURFACE_MODES = ("realistic", "flat_topography", "aquaplanet")
TOPOGRAPHIES = ("real", "flat")
BASIC_STATES = ("observed", "observed_sealevel", "zonal_mean", "landsea_contrast")

# Legacy presets: surface_mode -> (topography, basic_state). Explicit
# `topography` / `basic_state` keys in the config override these.
_SURFACE_PRESETS = {
    "realistic":       ("real", "observed"),
    "flat_topography": ("flat", "observed_sealevel"),
    "aquaplanet":      ("flat", "zonal_mean"),
}


def resolve_surface(cfg):
    """
    Return (topography, basic_state) for a config.

    topography : "real" | "flat"   -- PHIS read by the model (flat = sea level)
    basic_state: "observed"          -- observed NCEP climatology (relaxation targets
                                        contain the observed stationary waves)
                 "observed_sealevel" -- observed asymmetries, but ln(ps)/surface T at
                                        sea level (balanced with PHIS = 0)
                 "zonal_mean"        -- sea-level state, zonally averaged
                 "landsea_contrast"  -- zonal_mean + land-masked observed T asymmetry
                                        in the temperature relaxation target
    """
    preset = cfg.get("surface_mode") or "realistic"
    if preset not in SURFACE_MODES:
        raise ValueError(f"Unknown surface_mode '{preset}'. Choose from {SURFACE_MODES}.")
    topography, basic_state = _SURFACE_PRESETS[preset]
    topography = cfg.get("topography") or topography
    basic_state = cfg.get("basic_state") or basic_state
    if topography not in TOPOGRAPHIES:
        raise ValueError(f"Unknown topography '{topography}'. Choose from {TOPOGRAPHIES}.")
    if basic_state not in BASIC_STATES:
        raise ValueError(f"Unknown basic_state '{basic_state}'. Choose from {BASIC_STATES}.")
    if topography == "flat" and basic_state == "observed":
        raise ValueError(
            "topography=flat with basic_state=observed would pair PHIS = 0 with the observed "
            "surface pressure (which carries the mountain signal), leaving the basic state "
            "out of balance. Use basic_state=observed_sealevel (or zonal_mean / landsea_contrast).")
    return topography, basic_state


def topography_file(cfg):
    """Filename (inside the preprocess dir) of the PHIS the model should read."""
    topography, _ = resolve_surface(cfg)
    return "topog_flat.spectral.pt" if topography == "flat" else "topog.spectral.pt"


def build_preprocess_path(cfg):
    if cfg.get("preprocess_path_override"):
        path = cfg["preprocess_path_override"]
    else:
        zw, kmax = cfg["zw"], cfg["kmax"]
        season = cfg["season"].upper()
        y0, y1 = cfg["start_year"], cfg["end_year"]
        path = os.path.join(cfg["preprocess_root"],
                            f"preprocess__zw_{zw}__kmax_{kmax}_{season}_{y0}-{y1}")

    # Non-observed basic states get their own sibling directory so the shared
    # observed preprocess files are never overwritten. With real topography the
    # surface pressure and every sigma-level field must sit on the terrain (the
    # sea-level state mapped onto terrain-following levels), so that needs its own
    # directory ("_realtopo"); flat topography uses the plain sea-level state.
    # Idempotent: a path that already carries the suffix is not suffixed twice.
    topography, basic_state = resolve_surface(cfg)
    if basic_state != "observed":
        suffix = "_" + basic_state + ("_realtopo" if topography == "real" else "")
        if not path.rstrip("/").endswith(suffix):
            path = path.rstrip("/") + suffix
    return path


def build_experiment_path(cfg):
    return os.path.join(cfg["experiment_root"], cfg["experiment_name"])
