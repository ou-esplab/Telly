#!/usr/bin/env python
"""
Pipeline wrapper: runs any subset of the 4 ATM pipeline stages in order,
stopping at the first failure. Shared by the command line (this file's own
CLI) and the notebook UI (tools/configure_and_run_ui.py imports
run_pipeline_stages() directly), so there is exactly one place that knows
how to chain the stages together.

Usage:
    python scripts/run_pipeline.py --config config/experiments/my_exp.yaml
    python scripts/run_pipeline.py --config <path> --stages postprocess plot_results
    python scripts/run_pipeline.py --config <path> --screen

--stages accepts any subset of preprocess/run_model/postprocess/plot_results,
in any order -- they always execute in the canonical pipeline order regardless
of the order given. Default (omitted) is all 4.

--screen launches this same script, without --screen, detached in a `screen`
session (session name atm_<experiment_name>, log next to the config file) so
a long run survives closing the terminal/notebook.

--force confirms a cold_start=true wipe of an existing, non-empty experiment
directory (forwarded to 02_run_model.py's own --force) -- without it, that
stage refuses to delete real output rather than silently doing so.
"""

import argparse
import os
import shlex
import subprocess
import sys

from _config import load_config

STAGE_SCRIPTS = {
    "preprocess": "01_preprocess.py",
    "run_model": "02_run_model.py",
    "postprocess": "03_postprocess.py",
    "plot_results": "04_plot_results.py",
}
ALL_STAGES = list(STAGE_SCRIPTS)


def run_pipeline_stages(config_path, project_root, stages=None, print_fn=print, force=False):
    """
    Runs the given pipeline stages (default: all 4) in canonical order,
    stopping at the first failure. Returns True if every requested stage
    succeeded, False otherwise.

    force, if True, is passed through as --force to the run_model stage only
    -- confirms a cold_start=true wipe of an existing, non-empty experiment
    directory (see 02_run_model.py). Stages other than run_model don't accept
    --force and don't need it.
    """
    requested = stages if stages else ALL_STAGES
    ordered = [s for s in ALL_STAGES if s in requested]

    for stage in ordered:
        script = STAGE_SCRIPTS[stage]
        cmd = [sys.executable, os.path.join(project_root, "scripts", script),
               "--config", os.path.abspath(config_path)]
        if stage == "run_model" and force:
            cmd.append("--force")
        print_fn(f"--- Running {script} ---")
        result = subprocess.run(cmd, cwd=os.path.join(project_root, "scripts"),
                                 capture_output=True, text=True)
        print_fn(result.stdout)
        if result.returncode != 0:
            print_fn(result.stderr)
            print_fn(f"FAILED at {script} (exit {result.returncode}) — stopping.")
            return False
    print_fn("Pipeline complete.")
    return True


def _launch_screen(config_path, project_root, stages, force=False):
    cfg = load_config(config_path)
    experiment_name = cfg["experiment_name"]
    session_name = f"atm_{experiment_name}"
    config_dir = os.path.dirname(os.path.abspath(config_path))
    log_path = os.path.join(config_dir, f"{experiment_name}_pipeline.log")

    inner_cmd = [sys.executable, os.path.abspath(__file__),
                 "--config", os.path.abspath(config_path),
                 "--stages", *stages]
    if force:
        inner_cmd.append("--force")
    inner_cmd_str = " ".join(shlex.quote(part) for part in inner_cmd)
    shell_line = f"{inner_cmd_str} > {shlex.quote(log_path)} 2>&1"
    screen_cmd = ["screen", "-dmS", session_name, "bash", "-c", shell_line]

    try:
        subprocess.run(screen_cmd, cwd=os.path.join(project_root, "scripts"), check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"ERROR: failed to launch screen session: {e}")
        return False

    print(f"Submitted to screen session '{session_name}' -- stages run in the "
          "background, stopping at the first failure.")
    print(f"  Check progress:  screen -r {session_name}")
    print(f"  View the log:    tail -f {log_path}")
    print("  (detach from an attached screen session with Ctrl-A then D -- "
          "does not stop the run)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the ATM pipeline (any subset of its 4 stages).")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--stages", nargs="+", choices=ALL_STAGES, default=None,
                         help="Which stages to run (any order; always executed in pipeline "
                              "order). Default: all 4.")
    parser.add_argument("--screen", action="store_true",
                         help="Run detached in a screen session instead of blocking this shell.")
    parser.add_argument("--force", action="store_true",
                         help="Confirm a cold_start=true wipe of an existing, non-empty "
                              "experiment directory (forwarded to the run_model stage).")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stages = args.stages if args.stages else ALL_STAGES

    if args.screen:
        ok = _launch_screen(args.config, project_root, stages, force=args.force)
    else:
        ok = run_pipeline_stages(args.config, project_root, stages=stages, force=args.force)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
