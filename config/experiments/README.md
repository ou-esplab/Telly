# Experiment Configs

Place one YAML config file per experiment here. Copy from `../examples/` as a starting point.

Every config must set `experiment_name` explicitly to the exact directory name the
experiment's output should have under `experiment_root`. There is no automatic
derivation from `zw`/`season`/`heating_name`/years — set it to whatever you want,
and that's what gets created (e.g. `T63L26_DJF_mysensitivity_1999-2020`, or
`AC_mywarmrun` for a `gamma_ac` experiment).

See `../../README.md` for full parameter documentation.
