"""
Shared config-loading helper for the ATM workflow scripts.

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
