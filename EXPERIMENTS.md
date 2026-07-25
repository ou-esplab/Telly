# ATM Experiment Inventory

All experiments use **T63L26** resolution (zw=63, kmax=26, 96×192 Gaussian grid).

---

## Repository Layout

```
Atmospheric-Teleconnection-Model-main/   ← code lives here
│
├── EXPERIMENTS.md              ← this file
├── README.md
│
├── scripts/                    ← 4-step pipeline
│   ├── _config.py               ← shared load_config/build_preprocess_path/build_experiment_path
│   ├── _preprocess_common.py    ← shared grid/upsample helpers (extracted from 01_preprocess.py
│   │                               so generate_shape_scale.py can reuse them)
│   ├── 01_preprocess.py
│   ├── 02_run_model.py
│   ├── 03_postprocess.py
│   ├── 04_plot_results.py
│   ├── generate_heating.py      ← friendly wrapper over 01_preprocess.py --heating-only
│   ├── generate_shape_scale.py  ← gamma-distribution shape/scale fitting, incl. composite-years
│   │                               (e.g. El Nino) fits — ported from a notebook, see below
│   └── smoke_test.sh            ← cheap end-to-end pipeline check, see "Smoke Test" below
│
├── tools/                        ← Jupyter/ipywidgets UI, see "Tools" below
│   └── Configure_and_Run_Experiment.ipynb
│
├── config/
│   ├── defaults.yaml            ← values shared by every experiment config, see below
│   ├── examples/               ← documented YAML templates
│   │   ├── fixed_season_DJF_example.yaml
│   │   ├── gamma_ac_example.yaml
│   │   ├── smoke_test_fixed_season.yaml   ← used by scripts/smoke_test.sh, not a real experiment
│   │   └── smoke_test_gamma_ac.yaml       ← used by scripts/smoke_test.sh, not a real experiment
│   └── experiments/            ← one YAML per experiment
│       ├── T63L26_DJF_*.yaml   (9 DJF configs)
│       ├── T63L26_JJA_*.yaml   (9 JJA configs)
│       └── AC_*.yaml           (3 Gamma_AC configs)
│
├── FixedSeason_Model/          ← fixed-season model variant
│   ├── subs1_utils.py          ← core physics (imported by scripts); includes press_to_sig
│   └── reference_notebooks/    ← original/reference-only, not invoked by scripts/
│       ├── preprocess.ipynb, RunModel.beta.ipynb, PressureInterpMetPy.ipynb,
│       ├── plotResults.ipynb, heatingprofilefigureS8.ipynb, FigureS7.jpg
│       ├── RunModel.PrescribedMean.ipynb   ← unwired "strongly prescribed mean" variant
│       └── tropical_heating_weight_exploration.ipynb
│
├── Gamma_AC_Model/             ← annual-cycle stochastic model variant
│   ├── subs1_utils.py          ← same physics + latent_heat_release()
│   ├── RunModel.Gamma.py       ← called by 02_run_model.py (now takes --datapath/
│   │                              --prepath/--zw/--kmax/--tl from the config; no
│   │                              longer relies on hardcoded paths/resolution)
│   ├── RunModel.Gamma-noheating.py
│   ├── PressureInterpMetPy.py  ← called by 03_postprocess.py
│   ├── PressureInterpMetPy-Year.py
│   └── reference_notebooks/    ← original/reference-only, not invoked by scripts/
│       ├── preprocess_gamma.ipynb, preprocess.Gamma_heating.ipynb
│       ├── RunModel.Gamma.ipynb, RunModel.Gamma-noheating.ipynb
│       ├── PressureInterpMetPy.ipynb, PressureInterpMetPy_AC_Test_2027-2070.ipynb
│       ├── model_output_test.ipynb
│       └── runmodel.sh, postprocess.sh   ← superseded manual invocation examples
│
├── Postprocess/                ← standalone postprocess notebooks (reference)
│   ├── PressureInterpMetPy.ipynb
│   ├── SeaLevelPressure.ipynb
│   └── model_output_*.ipynb
│
└── Environments/               ← conda env files per platform
    ├── agcm_environment.yml
    └── agcm_environment_linux.yml  (etc.)
```

**Note:** `MultiThread_Model/old/` and the top-level `old/` directory (superseded code and empty leftover dirs) were removed as part of a repo cleanup pass. Reference-only notebooks that were previously mixed in at the top level of each model directory now live under `reference_notebooks/` so it's clear at a glance which files are actually invoked by `scripts/*.py` (listed above them) versus kept for reference. Two correctness fixes landed in the same pass: `press_to_sig` (needed by FixedSeason wind preprocessing) was restored into `FixedSeason_Model/subs1_utils.py`, and `RunModel.Gamma.py`'s previously-hardcoded output/preprocess paths and resolution now come from the YAML config via `scripts/02_run_model.py`. A later pass renamed `MultiThread_Model/` to `FixedSeason_Model/` and the `model_type`/`model_subtype` config values (`multithreaded`→`fixed_season`, `beta`→`weakly_prescribed_mean`, `prescribed_mean`→`strongly_prescribed_mean`) since the old names didn't reflect what the models do — `MultiThread_Model` had no threading/multiprocessing code at all (that lives in `Gamma_AC_Model` instead).

---

## Data Layout: AGCM vs. AGCM_Experiments

All data lives outside the repository under two top-level directories on
`/data/esplab/kpegion/projects/`:

```
/data/esplab/kpegion/projects/
├── AGCM/                          ← INPUTS to the model
│   ├── MultiThread_Model/
│   │   ├── input_data/            ← raw observational/reanalysis source files
│   │   │   ├── *.nc               ← NCEP, ERA5, CMAP climatology files
│   │   │   └── ...
│   │   ├── preprocess__zw_63__kmax_26/              ← JJA preprocess files
│   │   ├── preprocess__zw_63__kmax_26_v2/           ← older DJF preprocess (unused)
│   │   └── preprocess__zw_63__kmax_26_v2_DJF_1999-2020/  ← DJF preprocess files
│   └── AnnualCycle/               ← Gamma_AC preprocess files
│       ├── topog.spectral.pt
│       ├── temp.spectral.pt
│       ├── lnps.spectral.pt
│       ├── vortsig / divsig / tsig / usig / vsig *.pt
│       ├── shapeAC.pt, scaleAC.pt           ← gamma params for AC_Test (control)
│       ├── shapeAC_Warm.pt, scaleAC_Warm.pt ← gamma params for AC_warm, fit to composited
│       │                                       El Nino precipitation; NOT selectable by the
│       │                                       current scripts/02_run_model.py pipeline for a
│       │                                       fresh run — see note below
│       └── shape_noheating.pt, scale_noheating.pt ← used by RunModel.Gamma-noheating.py only
│
└── AGCM_Experiments/              ← OUTPUTS from the model
    ├── T63L26_DJF_1999-2020/
    ├── T63L26_DJF_ALL_1999-2020/
    ├── ...                        ← one subdirectory per experiment
    ├── AC_Test/
    ├── AC_warm/
    └── AC_noheating/
```

### What goes where and why

**`AGCM/` — inputs (read-only during a run)**

| Subdirectory | Contents | Generated by |
|---|---|---|
| `MultiThread_Model/input_data/` | Raw reanalysis/obs `.nc` files (NCEP, ERA5, CMAP) | Downloaded once; not regenerated |
| `MultiThread_Model/preprocess__*/` | Regridded, spectrally transformed `.pt` tensors ready for the model: topography, climatological winds/temperature/pressure, heating anomaly | `scripts/01_preprocess.py` |
| `AnnualCycle/` | Same as above but for the daily-varying annual cycle; also gamma distribution shape/scale parameters | `scripts/01_preprocess.py` (Gamma_AC path) |

Multiple experiments can share the same preprocess directory — all the background
state files (topography, climatological winds, temperature, surface pressure) are
identical across sensitivity runs that differ only in heating.  Only the
`heat*.pt` file changes.

**`AGCM_Experiments/` — outputs (written during a run)**

Each experiment gets its own subdirectory named after the config parameters.
Inside that directory:

| File pattern | Contents | Written by |
|---|---|---|
| `{var}_{date_start}_{date_end}.nc` | Daily model output in sigma-level coordinates (one 30-day chunk per file) | `02_run_model.py` step 2 (via `postprocessing()` in subs1_utils) |
| `lnps_{date_start}_{date_end}.nc` | Log surface pressure chunks (same 30-day cadence) | Same |
| `{var}mn_{date_start}_{date_end}.nc` | 30-day mean fields | Same |
| `{name}.spectral.pt` | Restart spectral-coefficient tensors (`zmn1`, `zmn2`, etc.) | `02_run_model.py` at end of run |
| `{var}_Pressure_days_1-{N}.nc` | Variables interpolated to pressure levels (850/500/300/200 hPa) | `03_postprocess.py` |
| `sealevelpressure_days_1-{N}.nc` | Sea-level pressure (if `compute_slp: true`) | `03_postprocess.py` |
| `figures/{var}{level}_{mean\|diff}.png` | Time-mean maps and anomaly maps vs. control | `04_plot_results.py` |

The restart `.pt` files are overwritten each time `02_run_model.py` completes,
so they always reflect the end of the most recent segment run.

---

## How to Run an Experiment

```bash
conda activate agcm_environment
cd /home/kpegion/projects/Atmospheric-Teleconnection-Model-main

# Step 1: Generate preprocess files (skip if they already exist)
python scripts/01_preprocess.py --config config/experiments/<name>.yaml

# Step 2: Run the model (set cold_start / toffset in YAML before running)
python scripts/02_run_model.py --config config/experiments/<name>.yaml

# Step 3: Interpolate to pressure levels
python scripts/03_postprocess.py --config config/experiments/<name>.yaml

# Step 4: Make standard figure set
python scripts/04_plot_results.py --config config/experiments/<name>.yaml
```

To **extend** an existing experiment: edit the config YAML, increase
`run_length_days`, set `cold_start: false`, and set `toffset` to the number of
days already completed, then re-run step 2 (and steps 3–4 afterward).

---

## Tools

`tools/Configure_and_Run_Experiment.ipynb` (needs `jupyterlab`/
`ipywidgets`, both in `Environments/agcm_environment.yml`) wraps the whole
command-line workflow above — generating input files *and* running the
pipeline — in a single form UI. No hand-editing YAML or the command line
required. Open it in JupyterLab from anywhere inside the repo (it locates
the project root automatically).

Every default in the notebook traces back to a real, verified control
experiment: `fixed_season` defaults to `T63L26_JJA_1979-2023`
(`config/experiments/T63L26_JJA_1979-2023.yaml`); `gamma_ac` defaults to
`AC_Test` (`config/experiments/AC_Test.yaml`, `shapeAC.pt`/`scaleAC.pt`).
Switching **Model** swaps in that model's control defaults and its matching
generation panel below the top fields:

- **`fixed_season`** shows a *Generate Heating File* panel. Its "Heating
  source" dropdown defaults to "Use control default (JJA 1979-2023)" — the
  preprocess directory already has that file, so nothing needs generating.
  Choosing "Generate new: custom file / from CCA / from CESM2 / from ERA5"
  reveals the matching input field and a Generate button, which wraps
  `scripts/generate_heating.py` (itself a thin wrapper over
  `01_preprocess.py --heating-only` — no new science, just a friendlier
  interface). Generation writes directly into the **Heating name**/
  **Preprocess dir** fields already set above, so there's nothing to retype
  before running.
- **`gamma_ac`** shows a *Generate Shape/Scale Files* panel — these are
  what the model actually uses for its daily stochastic heating draw
  (`heat.ggrid_*.pt` for `gamma_ac` is diagnostic-only, tucked into
  Advanced settings). Its dropdown defaults to "Use control default
  (shapeAC.pt / scaleAC.pt)". Choosing "Fit new: Control period" or
  "Fit new: Composite (e.g. El Nino)" or "Generate: No heating (zero)"
  wraps `scripts/generate_shape_scale.py`. Fit a plain climatology
  ("Control period"), a composite from event-year windows you specify
  (e.g. real El Nino years, the same technique behind `AC_warm`'s
  `shapeAC_Warm.pt`/`scaleAC_Warm.pt`), or generate an explicit all-zero
  pair. This fitting logic was ported from
  `Gamma_AC_Model/reference_notebooks/preprocess.Gamma_heating.ipynb`
  (previously manually-run-cells-only, no reusable function existed) —
  verified to reproduce the real `shapeAC_Warm.pt`/`scaleAC_Warm.pt`
  **exactly** (max abs difference 0.0) when given the same El Nino windows
  and date range. Composite windows must each span a full annual cycle
  (365/366 days, e.g. Jul-1-to-Jun-30) so the day-of-year composite has
  complete coverage — the tool validates this and raises a clear error
  rather than silently producing a partial-year result. On success, the
  **Shape file**/**Scale file** fields below auto-populate with the
  freshly generated `shape_<name>.pt`/`scale_<name>.pt`, so there's nothing
  to retype before running.

Below the generation panel, curated widgets cover the rest of the config
fields that vary per experiment (your own `experiment_root`/
`experiment_name`, run length, cold_start/toffset, control experiment, plot
vars), with the rarely-touched advanced fields (zw, kmax, chunk size, the
diagnostic `gamma_ac` heating file, legacy heating-filename overrides,
etc.) collapsed and defaulted from `config/defaults.yaml`. A "Build Config"
button writes the YAML; "Run Pipeline" then runs all 4 steps in sequence,
stopping at the first failure. If `cold_start` is checked and the target
experiment directory already exists, you're shown an explicit confirmation
button before anything is deleted.

You point `experiment_root` at your own directory — you won't have write
access to the instructor's production experiment data, so there's no
separate sandboxing logic needed beyond that.

---

## Smoke Test

```bash
conda activate agcm_environment
bash scripts/smoke_test.sh
```

Runs a cheap (2-3 simulated day) end-to-end check of all 4 steps for both
`fixed_season` and `gamma_ac`, reusing existing preprocess directories and
writing to `/tmp/atm_smoke_test` (never real experiment data). Checks that
restart tensors, raw netCDF, pressure-interpolated netCDF, and at least one
figure PNG all exist and are non-empty. Not a numerical-correctness check —
just confirms the pipeline didn't crash and produced files.

The first real run of this smoke test immediately found three bugs that had
apparently never been exercised by the config-driven pipeline before:
`01_preprocess.py` wasn't honoring `preprocess_path_override` (only
`02_run_model.py`/`03_postprocess.py` were), both `subs1_utils.py` copies'
`postprocessing()` built output filenames via raw string concatenation that
silently required a trailing slash on `datapath` (so netCDF chunk output
landed one directory up with a mangled name instead of erroring), and
`Gamma_AC_Model/PressureInterpMetPy.py` hardcoded `zw`/`kmax` and guessed its
own datapath from OS platform detection rather than the config's
`experiment_root`. All three are fixed; `build_preprocess_path` is now
shared via `scripts/_config.py` (not duplicated per-script) specifically to
stop this class of bug from being able to drift out of sync again.

Step 4 was later ported from ProPlot (unmaintained, incompatible with
Python ≥3.12) to plain matplotlib + cartopy — same map/colorbar/gridline
output, no functional change. Verified against both a sequential colormap
(time-mean plots) and a diverging one (difference plots). Also fixed a
latent crash in `04_plot_results.py`: a pressure level with no valid data
anywhere (e.g. entirely below ground for a very short run) made
`np.arange`'s step size computation blow up instead of skipping that
plot with a warning.

Step 1 originally had to be skipped for `gamma_ac` in this test — fixed:
`01_preprocess.py`'s "already complete" check required
`heat.ggrid_{heating_name}.pt` to exist for every model type, but that file
is never loaded by `RunModel.Gamma.py`/`RunModel.Gamma-noheating.py` (the
model's actual heating is entirely the stochastic draw from
`shapeAC.pt`/`scaleAC.pt` computed in `latent_heat_release()`). The
completeness check now only requires that file for `fixed_season`, which
does load it every timestep. The heating-generation code itself
(`gamma_preprocess_heating()`) is unchanged — still available via
`--force`/`--heating-only` for anyone who wants that diagnostic file — its
docstring was corrected since it previously (incorrectly) claimed the file
"acts as a background forcing."

---

## Preprocess Directories

| Alias | Path on disk | Used by |
|-------|-------------|---------|
| `v2_DJF` | `/data/esplab/kpegion/projects/AGCM/MultiThread_Model/preprocess__zw_63__kmax_26_v2_DJF_1999-2020` | All DJF experiments |
| `JJA` | `/data/esplab/kpegion/projects/AGCM/MultiThread_Model/preprocess__zw_63__kmax_26` | All JJA experiments |
| `AnnualCycle` | `/data/esplab/kpegion/projects/AGCM/AnnualCycle` | All Gamma_AC experiments |

---

## DJF Experiments (FixedSeason_Model, 1999–2020 climatology)

Data root: `/data/esplab/kpegion/projects/AGCM_Experiments/`
Preprocess: `v2_DJF` directory above.
Config files: `config/experiments/T63L26_DJF_*.yaml`

| Experiment dir | Heating file | Days run | Postprocessed output |
|----------------|--------------|----------|----------------------|
| `T63L26_DJF_1999-2020` | `heat_DJF_1999-2020.ggrid.pt` | **720** | `uvel_Pressure_days_1-720.nc`, `vvel_Pressure_days_1-720.nc` |
| `T63L26_DJF_ALL_1999-2020` | `heat_DJF_1999-2020_ALL.ggrid.pt` | **720** | `uvel_Pressure_days_1-720.nc`, `vvel_Pressure_days_1-720.nc` |
| `T63L26_DJF_ALL2_1999-2020` | `heat_DJF_1999-2020_ALL2.ggrid.pt` | **720** | `vvel_Pressure_days_1-720.nc` (uvel at days_1-60 only) |
| `T63L26_DJF_IOPAC_1999-2020` | `heat_DJF_1999-2020_IOPAC.ggrid.pt` | **720** | `uvel_Pressure_days_1-720.nc`, `vvel_Pressure_days_1-60.nc` + `days_1-720.nc` |
| `T63L26_DJF_IOPACneg_1999-2020` | `heat_DJF_1999-2020_IOPACneg.ggrid.pt` | **60** | `vvel_Pressure_days_1-60.nc` |
| `T63L26_DJF_PACComp2_1999-2020` | `heat_DJF_1999-2020_PACComp2.ggrid.pt` | **720** | `uvel_Pressure_days_1-720.nc`, `vvel_Pressure_days_1-60.nc` + `days_1-720.nc` |
| `T63L26_DJF_SAComp1_1999-2020` | `heat_DJF_1999-2020_SAComp1.ggrid.pt` | **720** | `uvel_Pressure_days_1-720.nc`, `vvel_Pressure_days_1-60.nc` + `days_1-720.nc` |
| `T63L26_DJF_SAComp1E_1999-2020` | `heat_DJF_1999-2020_SAComp1E.ggrid.pt` | **720** | `uvel_Pressure_days_1-720.nc` + `days_1-60.nc`, `vvel_Pressure_days_1-720.nc` + `days_1-60.nc` |
| `T63L26_DJF_SAComp2_1999-2020` | `heat_DJF_1999-2020_SAComp2.ggrid.pt` | **720** | `uvel_Pressure_days_1-720.nc`, `vvel_Pressure_days_1-60.nc` + `days_1-720.nc` |

**Control experiment**: `T63L26_DJF_1999-2020` (no anomalous heating — zero heat file).

**Notes on postprocessed output**: Several experiments have stale partial files
(e.g., `_days_1-60.nc`) left from earlier test runs alongside the full
`_days_1-720.nc` files. These duplicates are harmless but can be deleted.

### To extend a DJF experiment to more days

```bash
# Edit the config: set run_length_days to the new total, cold_start: false,
# toffset: 720 (days already run).
python scripts/02_run_model.py --config config/experiments/T63L26_DJF_SAComp1_1999-2020.yaml
```

---

## JJA Experiments (FixedSeason_Model, 1979–2023 climatology)

Data root: `/data/esplab/kpegion/projects/AGCM_Experiments/`
Preprocess: `JJA` directory above (no year suffix in dir name).
Config files: `config/experiments/T63L26_JJA_*.yaml`

| Experiment dir | Heating file | Days run | Postprocessed output |
|----------------|--------------|----------|----------------------|
| `T63L26_JJA_1979-2023` | `heat.ggrid_JJA_1979-2023.pt` | **720** | `geo_Pressure_days_1-720.nc`, `vvel_Pressure_days_1-720.nc` |
| `T63L26_JJA_Cluster1_ANA_1979-2023` | `heat.ggrid_Cluster1_ANA_1979-2023.pt` | **720** | `vvel_Pressure_days_1-720.nc` |
| `T63L26_JJA_Cluster2_ANA_1979-2023` | `heat.ggrid_Cluster2_ANA_1979-2023.pt` | **720** | `vvel_Pressure_days_1-720.nc` |
| `T63L26_JJA_Cluster3_ANA_1979-2023` | `heat.ggrid_Cluster3_ANA_1979-2023.pt` | **720** | `vvel_Pressure_days_1-720.nc` |
| `T63L26_JJA_Cluster4_ANA_1979-2023` | `heat.ggrid_Cluster4_ANA_1979-2023.pt` | **720** | `vvel_Pressure_days_1-720.nc` |
| `T63L26_JJA_Cluster1_SE` | `heat.ggrid_Cluster1_SE.pt` | **60** | `geo_Pressure_days_1-60.nc`, `vvel_Pressure_days_1-60.nc` |
| `T63L26_JJA_Cluster2_SE` | `heat.ggrid_Cluster2_SE.pt` | **60** | `geo_Pressure_days_1-60.nc`, `vvel_Pressure_days_1-60.nc` |
| `T63L26_JJA_Cluster3_SE` | `heat.ggrid_Cluster3_SE.pt` | **60** | `geo_Pressure_days_1-60.nc`, `vvel_Pressure_days_1-60.nc` |
| `T63L26_JJA_Cluster4_SE` | `heat.ggrid_Cluster4_SE.pt` | **60** | `geo_Pressure_days_1-60.nc`, `vvel_Pressure_days_1-60.nc` |

**Control experiment**: `T63L26_JJA_1979-2023`.

**ANA vs SE heating**: ANA = all-India precipitation composite heating
(1979–2023 period); SE = South East precipitation composite.

**Note on `*_SE` dir names**: All four `Cluster{1,2,3,4}_SE` on-disk directories
omit the year suffix (e.g. `T63L26_JJA_Cluster1_SE`, not `..._1979-2023`), unlike
their `*_ANA` siblings. This used to be a footgun — `02_run_model.py` previously
derived the output directory from a formula that got this wrong for 13 of the 21
experiment configs (not just the `*_SE` ones; also the DJF/JJA control runs and
all Gamma_AC experiments). Every config now sets `experiment_name` explicitly to
its real on-disk name (see `config/experiments/README.md`), so this is no longer
something a fresh run can get wrong.

---

## Gamma_AC Experiments (annual cycle model, 1979–2023 climatology)

Data root: `/data/esplab/kpegion/projects/AGCM_Experiments/`
Preprocess: `AnnualCycle` directory above.
Config files: `config/experiments/AC_*.yaml`

The Gamma_AC model integrates stochastic latent heating drawn each day from a
gamma distribution whose shape/scale parameters vary with day-of-year. Output
files use simulated dates starting at 1950-01-01 regardless of the climatology
period.  Output is post-processed annually by `PressureInterpMetPy.py` (one
`.nc` file per simulated year).

| Experiment dir | Preprocess files | Days run | Postprocessed output |
|----------------|-----------------|----------|----------------------|
| `AC_Test` | `shapeAC.pt`, `scaleAC.pt` | **54750** (150 yrs) | `geo_Pressure.nc` (all years); `geo_Pressure_days_1-6540.nc`, `_days_1-18240.nc`; ~122 annual geo files |
| `AC_warm` | `shapeAC_Warm.pt`, `scaleAC_Warm.pt` | **2160** (6 yrs) | `geo_Pressure_days_1-2160.nc`, `uvel_Pressure_days_1-2160.nc`, `vvel_Pressure_days_1-2160.nc` |
| `AC_noheating` | `shape_noheating.pt`, `scale_noheating.pt` | **0** (not yet run) | — |

**Control experiment**: `AC_Test` (default gamma distribution parameters).

**`shapeAC_Warm.pt`/`scaleAC_Warm.pt`**: gamma-distribution parameters fit
(via the same method-of-moments approach as the control) to precipitation
composited across three real El Nino episodes — Jul 2002–Jun 2003, Jul
2009–Jun 2010, Jul 2015–Jun 2016 — see
`Gamma_AC_Model/reference_notebooks/preprocess.Gamma_heating.ipynb`.

**Correction (previously this section incorrectly claimed `AC_warm` was
scientifically invalid and removed it from the repo)**: `RunModel.Gamma.py`
(the current `.py` script) hardcodes `prepath+'shapeAC.pt'`/`'scaleAC.pt'`
with no mechanism to select a different shape/scale file per experiment,
which led to the mistaken conclusion that the completed `AC_warm` run must
have used the wrong (control) parameters. That conclusion didn't check
`Gamma_AC_Model/reference_notebooks/RunModel.Gamma.ipynb`, which is what
actually produced the `AC_warm` output — its stored execution output
(real `FutureWarning` traces from `torch.load`, not just source code)
confirms `expname = 'AC_warm'`, `datapath` pointing at the real
`AGCM_Experiments/AC_warm/`, and `shape = torch.load(prepath+'shapeAC_Warm.pt')`
/ `scale = torch.load(prepath+'scaleAC_Warm.pt')` genuinely executing. The
completed run used the correct El Nino-specific parameters via a manually
edited notebook; the current `.py` script just never had that
per-experiment selection ported into it. `AC_warm`'s output and config
have been restored.

**Fixed**: `RunModel.Gamma.py` now takes `--shapefile`/`--scalefile`
(defaulting to `shapeAC.pt`/`scaleAC.pt`, same as before if omitted), and
`scripts/02_run_model.py` passes them from each config's
`shape_file_override`/`scale_file_override` — all three `AC_*.yaml` configs
now set these explicitly, so the config is the single source of truth for
which gamma parameters a run actually uses.

This also fixed a second, related gap found while making this change:
`run_gamma_ac()` in `scripts/02_run_model.py` always calls
`RunModel.Gamma.py`, never the separate `RunModel.Gamma-noheating.py`
script — so running `AC_noheating.yaml` through the automated pipeline
would previously have silently used the control parameters
(`shapeAC.pt`/`scaleAC.pt`) instead of `shape_noheating.pt`/
`scale_noheating.pt`, making it identical to `AC_Test` rather than an
actual no-heating run. `AC_noheating.yaml` now sets
`shape_file_override`/`scale_file_override` explicitly, so
`RunModel.Gamma-noheating.py` is no longer needed for a pipeline-driven
run of any of the three current experiments (it's still there, and still
directly runnable standalone, but superseded for this purpose).

**Known limitation (documented, not fixed retroactively): the shared
`AnnualCycle` preprocess directory's background-state climatology mixes two
different reference periods.** `gamma_preprocess_surface_pressure`/`_winds`
(surface pressure, u, v, T, q on pressure levels) slice reanalysis data to
`1994-01-01`–`2024-12-01`, a fixed 30-year window. `gamma_preprocess_temperature`
(surface temperature, used to build the 3-D background temperature field)
applies **no date slicing at all**, in the original reference notebook
(`Gamma_AC_Model/reference_notebooks/preprocess_gamma.ipynb`) or in every
copy of it found (including a scratch copy at
`/data/esplab/kpegion/scratch/gammamodel/preprocess_gamma.ipynb`, byte-diffed
against the repo's version — only the output path/filenames differ, not the
date logic) — it averages the *entire* NCEP/NCAR Reanalysis 1 period of
record at fetch time. That dataset starts in 1948 and is continuously
updated, and `temp.spectral.pt` in the real `AnnualCycle` directory is dated
April 1, 2025 — so the temperature climatology actually baked into `AC_Test`/
`AC_warm`/`AC_noheating`'s shared background state is effectively a
**~1948–2025 (~77-year) mean**, while surface pressure and winds reflect a
specific **1994–2024 (30-year) window**. This means the model's prescribed
background state combines a temperature field representing long-term-mean
conditions with pressure/wind fields representing a more recent, shorter
window — not a single mutually-consistent reference climate. Under any
long-term trend (e.g. warming), these two pieces of the background state
are not on equal footing.

`scripts/01_preprocess.py`'s `gamma_preprocess_temperature`/
`_surface_pressure`/`_winds` now all read `cfg["start_year"]`/
`cfg["end_year"]` consistently (previously only surface pressure and winds
did; temperature was hardcoded to no slicing regardless of any config), so
this specific inconsistency cannot recur in a newly-generated preprocess
directory. The existing `AnnualCycle` directory predates that fix and is
**not being regenerated** — doing so would require re-fetching from live
NCEP THREDDS endpoints and would change the actual climate `AC_Test`'s
existing 150-year run represents, which wasn't the point of this fix.
`AC_Test.yaml`/`AC_warm.yaml`/`AC_noheating.yaml`'s `start_year`/`end_year`
fields are set to `1994`/`2024` (accurate for 2 of the 3 background fields)
with a comment pointing back here — they're informational only for these
three configs, since the preprocess directory they point at already exists
and is complete, so `01_preprocess.py` never re-reads these two fields for
them.

**How to run Gamma_AC experiments via the existing script directly**:
```bash
cd Gamma_AC_Model
conda activate agcm_environment
# expstub is the part after "AC_"
python RunModel.Gamma.py --expname Test --toffset 0 --ichunk 5
# For a non-default experiment like AC_warm, pass the matching gamma files explicitly:
python RunModel.Gamma.py --expname warm --toffset 2160 --ichunk 1 \
    --shapefile shapeAC_Warm.pt --scalefile scaleAC_Warm.pt
```
Or use `02_run_model.py` with the YAML config (delegates to the script above,
reading `shape_file_override`/`scale_file_override` from the config).

**To run AC_noheating**:
```bash
python scripts/02_run_model.py --config config/experiments/AC_noheating.yaml
```

**Spin-up transient and why short runs look "wrong" next to `AC_Test`**: investigated live
comparing a fresh 30-day `gamma_ac` test run (`AC_Cntrl`, same `heating_name: Test` and
`shapeAC.pt`/`scaleAC.pt` as `AC_Test`) against `AC_Test` itself.

- **Cold start produces a real, large, reproducible transient.** Global-mean 850 hPa geopotential
  drops from ~14134 m² s⁻² on day 1 to ~13603 by day ~26-28 before leveling off — confirmed
  *identical* (to within noise) in `AC_Test`'s own days 1-30 and in `AC_Cntrl`'s days 1-30, so
  it's not specific to one config; it's the model adjusting from its prescribed (not dynamically
  balanced) initial/background state. This is why `spinup_days: 60` exists and why every
  Gamma_AC config here uses it — a mean that includes these days is not a climatology, it's
  dominated by the adjustment itself.
- **Past the transient, two independent runs still diverge — this is expected, not a bug.**
  Comparing `AC_Cntrl` and `AC_Test`'s own days 61-90 (identical config, same cold start,
  independent daily stochastic gamma-distributed heating draws) day-by-day: the two track closely
  through day ~66, then diverge to differences of -80 to -170 m² s⁻² by day ~80 — the same
  sensitive-dependence-on-forcing behavior that makes real weather unpredictable past 1-2 weeks.
  Quantified against `AC_Test`'s own post-spinup record: the std of 30-day global-mean-geo850
  block means across its full 150 years is ~154 m² s⁻² (range ~13443-13972 across 200 blocks) —
  so a single independent 30-day sample landing 70-170 m² s⁻² away from another is ordinary
  sampling variability, not evidence of a config or model error.
- **Practical implication**: a short `gamma_ac` run (tens of days) cannot be expected to match a
  long climatological run's (like `AC_Test`'s 150 years) mean closely, even with identical
  parameters and spin-up correctly excluded — only averaging over many more days (or many
  realizations) will converge toward the same climatology. A large `..._diff.png` between a short
  test run and `AC_Test` is expected, not necessarily a sign something's wrong.

**`AC_MJO`: composite MJO heating from real events, tiled as a repeating intraseasonal cycle**

Unlike the ENSO composites (`AC_ElNino`/`AC_LaNina`), MJO is a ~30-60 day intraseasonal oscillation,
not tied to specific years or calendar day-of-year — `generate_shape_scale.py`'s existing
`composite_windows` mechanism (day-of-year climatology across whole Jul-Jun year windows) doesn't
apply. Key enabler: `Gamma_AC_Model/RunModel.Gamma.py` indexes `shape[daynumber]`/`scale[daynumber]`
purely by real calendar day-of-year (0-364), with no other periodicity logic anywhere in the file —
so a *short, repeating* pattern tiled to fill 365 days cycles through the model automatically, with
**zero model-code changes**. Only the *generation* logic is new (`fit_gamma_shape_scale_mjo()` and
`_identify_mjo_onsets()`/`_mjo_phase()` in `scripts/generate_shape_scale.py`).

- **Data**: NOAA PSL's OMI index (`https://psl.noaa.gov/mjo/mjoindex/omi.1x.txt`), stored locally at
  `/data/esplab/shared/obs/indices/OMI/omi.1x.txt` (not committed — same convention as other
  observational input data). Whitespace-delimited, no header: `year month day PC1 PC2 amplitude`
  (amplitude = √(PC1²+PC2²)).
- **Phase formula, empirically validated, not assumed**: `phase = ((atan2(-PC1, PC2) in degrees) +
  157.5) // 45 % 8 + 1`. The sign convention (x=PC2, y=-PC1) matches NOAA's stated OMI/RMM
  relationship; the 157.5° sector-boundary offset (vs. the naive 180°) was found by grid search over
  sign/offset combinations, maximizing agreement against BOM's independently published RMM phases
  (`rmm.74toRealtime.txt`) on days both indices call amplitude>1: **92.5% agreement within ±1
  phase across 5055 overlapping days**, with a clean, symmetric, zero-centered error distribution —
  not an exact-match optimum fit to noise, and 157.5° (a half-sector shift from the naive guess)
  matches the physically-sensible convention of sectors being *centered* on cardinal directions
  rather than bounded by them.
- **Event onsets**: contiguous runs of `phase == target` & `amplitude > 1.0`; onset = first day of
  each run. Phase 3 (Indian Ocean-centered convection, the first phase built): 166 onsets across the
  1991-2026 OMI record (~4.7/year), median 60-day gap between onsets (matches expected MJO
  recurrence), no evidence of episode fragmentation.
- **Compositing**: for each onset, a window [onset-5, onset+15) (20 days) of CMORPH precip is
  extracted and stacked by *lag day* (not day-of-year) across all 127 usable events (1998-2024,
  CMORPH's available range) — 127 of 166 onsets had a complete window inside that range.
  Deliberately **skips** `fit_gamma_shape_scale()`'s monthly-resample-then-cubic-upsample smoothing
  step (built for the ENSO composites' slowly-varying *annual* cycle) — monthly-binning a 20-day
  cycle would leave ~1 bin per cycle and destroy the very structure being composited. Works at daily
  lag resolution throughout, method-of-moments fit per lag-day, then tiled (`np.tile`, truncated) to
  365 days.
- **Tile-wrap seam**: 365/20 isn't integer, so the last tile is truncated and day 364→day 0 has a
  discontinuity — checked directly: the wrap jump (~7 units in scale, at a sample tropical point) is
  *smaller* than the largest ordinary day-to-day jump elsewhere in the same cycle (~47 units), i.e.
  it blends into the composite's own natural noise level rather than standing out. No special
  seam-smoothing was needed for this window length.
- **Validation run**: `AC_MJO` (`config/experiments/AC_MJO.yaml`, `shape_file_override:
  shape_MJO_Phase3.pt`) run 90 days end-to-end — completed without error, physically plausible
  absolute fields. Its `..._diff.png` against `AC_Test` shows the same large magnitude as
  `AC_Cntrl`'s did — expected per the spin-up/short-sample section above (only 30 usable
  post-spinup days), **not** yet a meaningful estimate of any real MJO-forced teleconnection signal.
  A multi-year run (matching `AC_ElNino`/`AC_LaNina`'s scale) is needed before this comparison means
  anything scientifically; deliberately deferred to avoid 3-way resource contention with those two
  extensions already in progress.
- **Notebook UI**: `tools/configure_and_run_ui.py`'s shape/scale panel has a "Fit new: MJO
  composite" mode (phase, min amplitude, lag-before/after, OMI index path fields), calling
  `fit_gamma_shape_scale_mjo()` the same way the CLI does — verified via a stubbed-exec harness
  (correct widget visibility per mode, correct call arguments).
- **Deferred**: the other 7 MJO phases (only phase 3 built/validated so far).

---

## Misc / Test Directories

| Directory | Notes |
|-----------|-------|
| `TestT63L26_Dist` | Short test run; purpose unclear. Not documented with a config. |

---

## What Still Needs Running / Postprocessing

| Experiment | Issue |
|------------|-------|
| `T63L26_DJF_IOPACneg_1999-2020` | Only 60 days; needs extension to 720 days |
| `T63L26_JJA_Cluster*_SE` | Only 60 days; needs extension to 720 days |
| `T63L26_DJF_ALL2_1999-2020` | `uvel` pressure file covers only days 1-60; needs full 720-day postprocess |
| `T63L26_JJA_Cluster*_ANA_1979-2023` | Only `vvel` postprocessed; missing `uvel` and `geo` |
| `AC_noheating` | Not yet run |
| `AC_Test` | Only `geo` postprocessed (partially); `uvel` and `vvel` missing |
