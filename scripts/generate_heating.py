#!/usr/bin/env python
"""
Friendly wrapper for generating a heating file (fixed_season or gamma_ac)
without hand-writing a full experiment config. Does not reimplement any
science — writes a minimal scratch config and invokes
`python scripts/01_preprocess.py --config <scratch> --heating-only`
(an existing, already-tested flag) as a subprocess.

Usage (CLI):
    python scripts/generate_heating.py --model-type fixed_season \\
        --heating-source custom --heating-file /path/to/prebuilt_heat.pt \\
        --heating-name MyScenario --season DJF --start-year 1999 --end-year 2020 \\
        --zw 63 --kmax 26 --output-dir /path/to/preprocess/dir

Also importable directly:
    from generate_heating import generate_heating_file
"""

import argparse
import os
import subprocess
import sys
import tempfile

import yaml


def generate_heating_file(model_type, heating_source, heating_name, output_dir,
                           season=None, zw=63, kmax=26, start_year=None, end_year=None,
                           heating_file=None, cesm2_precip_file=None, cca_precip_file=None,
                           era5_precip_file=None, enso_warm_years=None, input_data_path=None):
    """
    Generate a single heat.ggrid_{heating_name}.pt file into output_dir.

    model_type    : "fixed_season" or "gamma_ac"
    heating_source: "custom" | "cca" | "cesm2" | "era5" (fixed_season);
                    "custom" or omitted (gamma_ac default ENSO-composite path)
    heating_name  : label -> output file heat.ggrid_{heating_name}.pt
    output_dir    : preprocess directory to write into (passed through as
                    preprocess_path_override, independent of any real
                    experiment's directory-naming convention)
    season        : required for fixed_season (DJF/JJA/MAM/SON), unused for gamma_ac
    start_year/end_year: required for fixed_season non-custom sources
    heating_file  : path to a pre-made heating tensor, for heating_source=custom
    cesm2_precip_file/cca_precip_file/era5_precip_file: source-specific precip files
    enso_warm_years: gamma_ac default-branch only, list of year strings
    input_data_path: falls back to config/defaults.yaml's value if not given

    Raises subprocess.CalledProcessError if generation fails.
    """
    if model_type not in ("fixed_season", "gamma_ac"):
        raise ValueError(f"model_type must be 'fixed_season' or 'gamma_ac', got {model_type!r}")
    if model_type == "fixed_season":
        # mt_preprocess_heating reads season/start_year/end_year unconditionally,
        # before it even checks heating_source — confirmed empirically, not just
        # from reading the code, so this is required for every source including
        # "custom", not just the non-custom branches.
        missing = [k for k, v in (("season", season), ("start_year", start_year),
                                   ("end_year", end_year)) if v is None]
        if missing:
            raise ValueError(
                f"fixed_season heating generation requires {missing} "
                "(mt_preprocess_heating reads these regardless of heating_source)."
            )

    cfg = {
        "model_type": model_type,
        "zw": zw,
        "kmax": kmax,
        "heating_name": heating_name,
        "heating_source": heating_source,
        "preprocess_path_override": output_dir,
        # season is required for fixed_season regardless of heating_source
        # (season_month_indices is called unconditionally); harmless if
        # gamma_ac ignores it.
        "season": season or "annual",
    }
    if start_year is not None:
        cfg["start_year"] = start_year
    if end_year is not None:
        cfg["end_year"] = end_year
    if heating_file is not None:
        cfg["heating_file"] = heating_file
    if cesm2_precip_file is not None:
        cfg["cesm2_precip_file"] = cesm2_precip_file
    if cca_precip_file is not None:
        cfg["cca_precip_file"] = cca_precip_file
    if era5_precip_file is not None:
        cfg["era5_precip_file"] = era5_precip_file
    if enso_warm_years is not None:
        cfg["enso_warm_years"] = enso_warm_years
    if input_data_path is not None:
        cfg["input_data_path"] = input_data_path

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(project_root, "scripts", "01_preprocess.py")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
        scratch_config = f.name

    try:
        cmd = [sys.executable, script, "--config", scratch_config, "--heating-only"]
        print(f"  Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=os.path.join(project_root, "scripts"))
    finally:
        os.unlink(scratch_config)

    outfile = os.path.join(output_dir, f"heat.ggrid_{heating_name}.pt")
    print(f"  Heating file: {outfile}")
    return outfile


def main():
    parser = argparse.ArgumentParser(description="Generate a heating file (fixed_season or gamma_ac)")
    parser.add_argument("--model-type", required=True, choices=["fixed_season", "gamma_ac"])
    parser.add_argument("--heating-source", required=True,
                         help="custom | cca | cesm2 | era5 (fixed_season); custom or omit for gamma_ac default")
    parser.add_argument("--heating-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--season", default=None, help="DJF | JJA | MAM | SON (required for fixed_season)")
    parser.add_argument("--zw", type=int, default=63)
    parser.add_argument("--kmax", type=int, default=26)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--heating-file", default=None, help="Pre-made heating tensor (heating_source=custom)")
    parser.add_argument("--cesm2-precip-file", default=None)
    parser.add_argument("--cca-precip-file", default=None)
    parser.add_argument("--era5-precip-file", default=None)
    parser.add_argument("--enso-warm-years", nargs="*", default=None, help="gamma_ac default path only")
    parser.add_argument("--input-data-path", default=None)
    args = parser.parse_args()

    generate_heating_file(
        model_type=args.model_type,
        heating_source=args.heating_source,
        heating_name=args.heating_name,
        output_dir=args.output_dir,
        season=args.season,
        zw=args.zw,
        kmax=args.kmax,
        start_year=args.start_year,
        end_year=args.end_year,
        heating_file=args.heating_file,
        cesm2_precip_file=args.cesm2_precip_file,
        cca_precip_file=args.cca_precip_file,
        era5_precip_file=args.era5_precip_file,
        enso_warm_years=args.enso_warm_years,
        input_data_path=args.input_data_path,
    )


if __name__ == "__main__":
    main()
