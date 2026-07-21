# Telly — Atmospheric Teleconnection Model

This repository is the OU ESPLab fork of the Atmospheric Teleconnection Model (ATM), a
simplified-physics atmospheric general circulation model for idealized climate dynamics studies.
It tracks the upstream model but adds a config-driven pipeline and a form-based Jupyter UI
(`student_tools/Configure_and_Run_Experiment.ipynb`) aimed at students running experiments without
hand-editing notebooks or YAML.

- **Original model repository**: https://github.com/jsb288/Atmospheric-Teleconnection-Model
- **Model paper**: Kirtman, B. P., and Coauthors, 2025: A Simplified-Physics Atmosphere General
  Circulation Model for Idealized Climate Dynamics Studies. *Bull. Amer. Meteor. Soc.*, 106,
  E2073–E2086, https://doi.org/10.1175/BAMS-D-24-0196.1.

If you use this model, please cite the paper above and refer to the original repository for the
underlying (non-fork-specific) model documentation.

## Getting Started

### 1. Get the code from GitHub

Clone the repository:

```bash
git clone git@github.com:ou-esplab/Telly.git
# or, over HTTPS:
git clone https://github.com/ou-esplab/Telly.git
```

If you don't use git, you can also click the green "<> Code" button on the
[repository page](https://github.com/ou-esplab/Telly) and choose "Download ZIP", then unzip the
folder. You can move the entire folder, but moving individual files within it may break scripts
that locate the project root relative to their own file path.

### 2. Set up the conda environment

`Environments/agcm_environment.yml` is the primary, actively-maintained environment spec, covering
all 4 pipeline steps:

```bash
conda env create -f Environments/agcm_environment.yml
```

`Environments/agcm_environment_{linux,mac,windows}.yml` are known-good snapshots frozen via `conda
env export` on a specific machine at a specific point in time; use one of those only as a fallback
reference if `agcm_environment.yml` doesn't resolve on your machine — their pinned build hashes make
them unlikely to solve on a different machine/date. See the
[Conda User Guide](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
for more on creating an environment from a yml file.

If the yml file gives you trouble, this sequence is a reasonable fallback:

```bash
conda install -n agcm_environment xarray pandas scipy netcdf4 metpy pyyaml matplotlib cartopy jupyter pytorch
conda install -n agcm_environment -c conda-forge xesmf
conda activate agcm_environment
pip3 install torch-harmonics==0.6.3
```

### 3. Run an experiment using the UI

Most users — especially students — should use
[`student_tools/Configure_and_Run_Experiment.ipynb`](student_tools/Configure_and_Run_Experiment.ipynb)
instead of running scripts by hand. Open it in JupyterLab from anywhere inside the repo (it locates
the project root automatically) and it wraps the whole workflow in a single form UI.

Every default in the notebook traces back to a real, verified control experiment. Switching
**Model** swaps in that model's control defaults and its matching generation panel below the top
fields:

- **`fixed_season`** shows a *Generate Heating File* panel. Its "Heating source" dropdown defaults
  to using the control heating file already present in the preprocess directory, so nothing needs
  generating. Choosing "Generate new: custom file / from CCA / from CESM2 / from ERA5" reveals the
  matching input field and a Generate button. Generation writes directly into the **Heating
  name**/**Preprocess dir** fields already set above, so there's nothing to retype before running.
- **`gamma_ac`** shows a *Generate Shape/Scale Files* panel — these are what the model actually
  uses for its daily stochastic heating draw. Its dropdown defaults to using the control
  shape/scale files. Choosing "Fit new: Control period", "Fit new: Composite (e.g. El Nino)", or
  "Generate: No heating (zero)" lets you fit a plain climatology, a composite from event-year
  windows you specify (e.g. real El Nino years), or generate an explicit all-zero pair. Composite
  windows must each span a full annual cycle (365/366 days, e.g. Jul-1-to-Jun-30) so the
  day-of-year composite has complete coverage — the tool validates this and raises a clear error
  rather than silently producing a partial-year result. On success, the **Shape file**/**Scale
  file** fields below auto-populate with the freshly generated files.

Below the generation panel, curated widgets cover the rest of the config fields that vary per
experiment (your own `experiment_root`/`experiment_name`, run length, cold_start/toffset, control
experiment, plot vars), with rarely-touched advanced fields collapsed and defaulted. A "Build
Config" button writes the YAML; "Run Pipeline" then runs all 4 steps in sequence, stopping at the
first failure. Check "Run in background (screen)" first if you want the run to survive closing the
notebook or losing your connection. If `cold_start` is checked and the target experiment directory
already exists, you're shown an explicit confirmation button before anything is deleted.

You should point `experiment_root` at your own directory rather than any shared/instructor
directory you may not have write access to.

### 4. Running the pipeline by hand (advanced)

If you need to run steps individually, or write your own experiment config:

```bash
conda activate agcm_environment
cd Telly   # or wherever you cloned the repo

# Step 1: Generate preprocess files (skip if they already exist)
python scripts/01_preprocess.py --config config/experiments/<name>.yaml

# Step 2: Run the model (set cold_start / toffset in YAML before running)
python scripts/02_run_model.py --config config/experiments/<name>.yaml

# Step 3: Interpolate to pressure levels
python scripts/03_postprocess.py --config config/experiments/<name>.yaml

# Step 4: Make standard figure set
python scripts/04_plot_results.py --config config/experiments/<name>.yaml
```

To **extend** an existing experiment: edit the config YAML, increase `run_length_days`, set
`cold_start: false`, and set `toffset` to the number of days already completed, then re-run step 2
(and steps 3–4 afterward).

### Repository layout

```
scripts/                    ← 4-step pipeline (01_preprocess.py … 04_plot_results.py),
                               plus generate_heating.py / generate_shape_scale.py helpers
                               and smoke_test.sh
student_tools/               ← Jupyter/ipywidgets UI (Configure_and_Run_Experiment.ipynb)
config/
├── defaults.yaml            ← values shared by every experiment config
├── examples/                ← documented YAML templates
└── experiments/             ← one YAML per experiment
FixedSeason_Model/            ← fixed-season model variant (core physics + reference notebooks)
Gamma_AC_Model/               ← annual-cycle stochastic model variant
Postprocess/                  ← standalone postprocess notebooks (reference)
Environments/                 ← conda env files per platform
```

Each model directory's `reference_notebooks/` subfolder holds the original, reference-only
notebooks this pipeline was ported from — they're not invoked by anything under `scripts/`.

### Smoke test

```bash
conda activate agcm_environment
bash scripts/smoke_test.sh
```

Runs a cheap (2-3 simulated day) end-to-end check of all 4 steps for both `fixed_season` and
`gamma_ac`, reusing existing preprocess directories and writing to `/tmp/atm_smoke_test` (never
real experiment data). Checks that restart tensors, raw netCDF, pressure-interpolated netCDF, and
at least one figure PNG all exist and are non-empty. Not a numerical-correctness check — just
confirms the pipeline didn't crash and produced files.

## Troubleshooting

For issues, questions, or concerns about the model itself, see the
[original repository](https://github.com/jsb288/Atmospheric-Teleconnection-Model) or contact Ben
Kirtman at bkirtman@miami.edu. For issues specific to this fork's pipeline or student tools, open
an issue on [this repository](https://github.com/ou-esplab/Telly).
