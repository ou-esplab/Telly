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
│   ├── 01_preprocess.py
│   ├── 02_run_model.py
│   ├── 03_postprocess.py
│   ├── 04_plot_results.py
│   └── smoke_test.sh            ← cheap end-to-end pipeline check, see "Smoke Test" below
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
│       └── AC_*.yaml           (2 Gamma_AC configs — AC_warm removed, see Gamma_AC section)
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
│       ├── shapeAC.pt, scaleAC.pt           ← gamma params for AC_Test
│       ├── shapeAC_Warm.pt, scaleAC_Warm.pt ← NOT currently loaded by any script (see note below)
│       └── shape_noheating.pt, scale_noheating.pt ← used by RunModel.Gamma-noheating.py only
│
└── AGCM_Experiments/              ← OUTPUTS from the model
    ├── T63L26_DJF_1999-2020/
    ├── T63L26_DJF_ALL_1999-2020/
    ├── ...                        ← one subdirectory per experiment
    ├── AC_Test/
    ├── AC_warm_SUSPECT_gamma_params/  ← removed from active use, see note below
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
| `AC_noheating` | `shape_noheating.pt`, `scale_noheating.pt` | **0** (not yet run) | — |

**Control experiment**: `AC_Test` (default gamma distribution parameters).

**`AC_warm` removed from active use**: `RunModel.Gamma.py` hardcodes
`prepath+'shapeAC.pt'`/`'scaleAC.pt'` with no mechanism to select a different
shape/scale file per experiment — `shapeAC_Warm.pt`/`scaleAC_Warm.pt` exist in
the preprocess directory but are never loaded by any current script. The
already-completed `AC_warm` run (2160 days) was therefore very likely
integrated with the *same* gamma parameters as the `AC_Test` control, not
distinct "warm" ones, despite its name and prior documentation here claiming
otherwise. Its output has been moved aside on disk to
`AC_warm_SUSPECT_gamma_params/` (not deleted, in case it's worth inspecting
later) and `config/experiments/AC_warm.yaml` removed from the repo. Re-running
a real "warm" experiment requires first adding a way for `RunModel.Gamma.py`
to select a non-default shape/scale file per experiment — not yet done.

**How to run Gamma_AC experiments via the existing script directly**:
```bash
cd Gamma_AC_Model
conda activate agcm_environment
# expstub is the part after "AC_"
python RunModel.Gamma.py --expname Test --toffset 0 --ichunk 5
```
Or use `02_run_model.py` with the YAML config (delegates to the script above).

**To run AC_noheating**:
```bash
python scripts/02_run_model.py --config config/experiments/AC_noheating.yaml
```

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
