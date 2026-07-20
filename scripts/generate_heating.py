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
                           cmap_precip_file=None, cmorph_precip_glob=None, anomaly_type=None,
                           enso_warm_years=None, input_data_path=None,
                           anomaly_years=None, anomaly_lat_min=None, anomaly_lat_max=None,
                           anomaly_lon_min=None, anomaly_lon_max=None):
    """
    Generate a single heat.ggrid_{heating_name}.pt file into output_dir.

    model_type    : "fixed_season" or "gamma_ac"
    heating_source: "custom" | "cca" | "cesm2" | "cmap" | "cmorph" (fixed_season);
                    "custom" or omitted (gamma_ac default ENSO-composite path)
    heating_name  : label -> output file heat.ggrid_{heating_name}.pt
    output_dir    : preprocess directory to write into (passed through as
                    preprocess_path_override, independent of any real
                    experiment's directory-naming convention)
    season        : required for fixed_season (DJF/JJA/MAM/SON), unused for gamma_ac
    start_year/end_year: required for fixed_season non-custom sources
    heating_file  : path to a pre-made heating tensor, for heating_source=custom
    cesm2_precip_file/cca_precip_file: source-specific precip ANOMALY files
        (climatology already removed upstream -- these are not raw precip)
    cmap_precip_file: single netCDF file (heating_source=cmap) -- raw monthly
        precip by default, or a precomputed anomaly if anomaly_type="precomputed"
    cmorph_precip_glob: glob pattern matching daily netCDF files
        (heating_source=cmorph), e.g. .../CMORPH_V1.0_ADJ_0.25deg-DLY_00Z_*.nc
    anomaly_type: "precomputed" (default -- file/glob already IS an anomaly) or
        "compute" (cmap/cmorph only -- subtract that dataset's own climatology,
        grouped by month for cmap / day-of-year for cmorph)
    enso_warm_years: gamma_ac default-branch only, list of year strings
    input_data_path: falls back to config/defaults.yaml's value if not given
    anomaly_years: fixed_season cesm2/cca/cmap/cmorph sources only -- list of
        year strings to composite the anomaly over (mirrors enso_warm_years'
        pattern for gamma_ac). Omit/None = use the anomaly as-is.
    anomaly_lat_min/anomaly_lat_max/anomaly_lon_min/anomaly_lon_max:
        fixed_season cesm2/cca/cmap/cmorph sources only -- optional region box
        (0-360 longitude convention) to mask the anomaly to before
        combining with climatology. All four or none.

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
    if cmap_precip_file is not None:
        cfg["cmap_precip_file"] = cmap_precip_file
    if cmorph_precip_glob is not None:
        cfg["cmorph_precip_glob"] = cmorph_precip_glob
    if anomaly_type is not None:
        cfg["anomaly_type"] = anomaly_type
    if enso_warm_years is not None:
        cfg["enso_warm_years"] = enso_warm_years
    if input_data_path is not None:
        cfg["input_data_path"] = input_data_path
    if anomaly_years is not None:
        cfg["anomaly_years"] = anomaly_years
    if anomaly_lat_min is not None:
        cfg["anomaly_lat_min"] = anomaly_lat_min
    if anomaly_lat_max is not None:
        cfg["anomaly_lat_max"] = anomaly_lat_max
    if anomaly_lon_min is not None:
        cfg["anomaly_lon_min"] = anomaly_lon_min
    if anomaly_lon_max is not None:
        cfg["anomaly_lon_max"] = anomaly_lon_max

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
                         help="custom | cca | cesm2 | cmap | cmorph (fixed_season); custom or omit for gamma_ac default")
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
    parser.add_argument("--cmap-precip-file", default=None)
    parser.add_argument("--cmorph-precip-glob", default=None)
    parser.add_argument("--anomaly-type", default=None, choices=["precomputed", "compute"],
                         help="cmap/cmorph only: 'compute' subtracts that dataset's own climatology")
    parser.add_argument("--enso-warm-years", nargs="*", default=None, help="gamma_ac default path only")
    parser.add_argument("--input-data-path", default=None)
    parser.add_argument("--anomaly-years", nargs="*", default=None,
                         help="fixed_season cesm2/cca/cmap/cmorph only: composite the anomaly over these years")
    parser.add_argument("--anomaly-lat-min", type=float, default=None)
    parser.add_argument("--anomaly-lat-max", type=float, default=None)
    parser.add_argument("--anomaly-lon-min", type=float, default=None, help="0-360 convention")
    parser.add_argument("--anomaly-lon-max", type=float, default=None, help="0-360 convention")
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
        cmap_precip_file=args.cmap_precip_file,
        cmorph_precip_glob=args.cmorph_precip_glob,
        anomaly_type=args.anomaly_type,
        enso_warm_years=args.enso_warm_years,
        input_data_path=args.input_data_path,
        anomaly_years=args.anomaly_years,
        anomaly_lat_min=args.anomaly_lat_min,
        anomaly_lat_max=args.anomaly_lat_max,
        anomaly_lon_min=args.anomaly_lon_min,
        anomaly_lon_max=args.anomaly_lon_max,
    )


if __name__ == "__main__":
    main()
