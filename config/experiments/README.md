# Experiment Configs

Place one YAML config file per experiment here. Copy from `../examples/` as a starting point.

Every config must set `experiment_name` explicitly to the exact directory name the
experiment's output should have under `experiment_root`. There is no automatic
derivation from `zw`/`season`/`heating_name`/years — set it to whatever you want,
and that's what gets created (e.g. `T63L26_DJF_mysensitivity_1999-2020`, or
`AC_mywarmrun` for a `gamma_ac` experiment).

`../defaults.yaml` supplies values shared by every current experiment (`zw`, `kmax`,
`chunk_size_days`, `compute_slp`, `experiment_root`, `heating_source`,
`input_data_path`, `plot_levels_hpa`, `postprocess_vars`, `pressure_levels_hpa`) —
loaded automatically by `scripts/_config.py:load_config()` and merged underneath
whatever this file sets. Your experiment config only needs to include a key if its
value differs from `../defaults.yaml`; anything you do set here always wins. The two
example templates in `../examples/` show every key (including the defaulted ones,
marked `[default]`) for reference, but a real experiment config should look like the
existing files in this directory — short, and only stating what's specific to it.

See `../../README.md` for full parameter documentation.
