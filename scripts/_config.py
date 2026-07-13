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


def build_preprocess_path(cfg):
    if cfg.get("preprocess_path_override"):
        return cfg["preprocess_path_override"]
    zw, kmax = cfg["zw"], cfg["kmax"]
    season = cfg["season"].upper()
    y0, y1 = cfg["start_year"], cfg["end_year"]
    return os.path.join(cfg["preprocess_root"],
                        f"preprocess__zw_{zw}__kmax_{kmax}_{season}_{y0}-{y1}")


def build_experiment_path(cfg):
    return os.path.join(cfg["experiment_root"], cfg["experiment_name"])
