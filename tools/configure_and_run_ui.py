"""
UI (widgets, config-building, pipeline-running) for the
Configure & Run Experiment notebook. Lives here rather than inline in the
notebook cell so it's editable with normal tools (real diffs, not notebook
JSON surgery) and easier to debug (a traceback points at a real file/line).
"""

import datetime
import glob
import re
import subprocess
import time
import ipywidgets as w
import yaml

from generate_heating import generate_heating_file
from generate_shape_scale import fit_gamma_shape_scale, zero_shape_scale
from run_pipeline import run_pipeline_stages
import os
import sys

from IPython.display import display, Image

_DUPLICATE_CLICK_WINDOW_S = 3.0


def _is_duplicate_click(last_click_state):
    # Guards a Generate button against a burst of rapid clicks -- ipywidgets
    # processes queued click messages one at a time, each running to full
    # completion (including resetting button.disabled back to False in a
    # finally block) before the next queued click is even dispatched. So a
    # second click sent before the button's disabled state has visually
    # rendered in the browser isn't blocked by disabling the button -- by
    # the time it actually runs, the first click has already finished and
    # re-enabled it. This time-based check catches that case regardless.
    now = time.time()
    if now - last_click_state[0] < _DUPLICATE_CLICK_WINDOW_S:
        return True
    last_click_state[0] = now
    return False



def build_and_display_ui(project_root):

    _LABEL_STYLE = {"description_width": "150px"}


    def _wide():
        # A fresh Layout instance per widget -- widgets given the *same* Layout
        # object share that object's traits, so toggling .layout.display on one
        # would silently toggle it on every other widget using it too.
        return w.Layout(width="420px")


    def _group_box():
        # Bordered/padded container so a generation panel's fields visually read
        # as belonging to its header, not as loose widgets floating below it.
        return w.Layout(border="1px solid #999", padding="8px", margin="4px 0px")


    # Real, verified control experiments — every default in this notebook
    # traces back to one of these two (config/experiments/T63L26_JJA_1979-2023.yaml,
    # config/experiments/AC_Test.yaml). run_length_days/cold_start/toffset/spinup_days
    # stay at short first-test values for both model types (AC_Test's real
    # run_length_days is 54750 -- ~150 simulated years -- not a sane UI default).
    FIXED_CONTROL_JJA = dict(
        season="JJA", heating_name="JJA_1979-2023", start_year=1979, end_year=2023,
        preprocess_path="/data/esplab/kpegion/projects/AGCM/MultiThread_Model/preprocess__zw_63__kmax_26",
        heating_file_override="",
    )
    # config/experiments/T63L26_DJF_1999-2020_control.yaml -- a separate, DJF-specific
    # preprocess directory (fixed_season's background climatology is season-specific,
    # not just year-range-specific -- mt_preprocess_temperature/_surface_pressure both
    # average over the chosen Season's months, so JJA and DJF can't share one directory).
    # Its real heat file predates the heat.ggrid_{heating_name}.pt naming convention,
    # hence heating_file_override.
    FIXED_CONTROL_DJF = dict(
        season="DJF", heating_name="control", start_year=1999, end_year=2020,
        preprocess_path="/data/esplab/kpegion/projects/AGCM/MultiThread_Model/preprocess__zw_63__kmax_26_v2_DJF_1999-2020",
        heating_file_override="heat_DJF_1999-2020.ggrid.pt",
    )
    # Kept as the notebook's initial (pre-any-selection) defaults -- JJA is
    # EXISTING_MODES[0], the existing-file sub-dropdown's own default value.
    FIXED_CONTROL = FIXED_CONTROL_JJA
    # start_year/end_year=1994/2024 -- best single approximation of what
    # actually built the real AC_Test/AC_warm/AC_noheating background-state
    # climatology, but NOT fully accurate: surface pressure and winds really
    # were sliced to 1994-2024, but temperature was never sliced at all (full
    # NCEP reanalysis period of record at fetch time, ~1948 through the Apr
    # 2025 generation date) -- a real inconsistency in the existing production
    # data, left as-is rather than regenerated. See EXPERIMENTS.md's Gamma_AC
    # section and AC_Test.yaml's comment for the full writeup.
    # scripts/01_preprocess.py's gamma_preprocess_* functions now all read
    # cfg start_year/end_year consistently, so this can't recur for a NEW
    # preprocess directory -- generating one from this notebook with these
    # defaults would build a properly self-consistent climatology.
    GAMMA_CONTROL = dict(
        heating_name="Test", start_year=1994, end_year=2024,
        preprocess_path="/data/esplab/kpegion/projects/AGCM/AnnualCycle",
    )
    # Instructor's own experiment_root (config/defaults.yaml) -- shown as a
    # concrete, working suggestion. You won't have write access there, so
    # leaving it as-is just fails with a permission error; change it to your own.
    SUGGESTED_EXPERIMENT_ROOT = "/data/esplab/kpegion/projects/AGCM_Experiments"

    # --- Curated (main) fields ---
    r_model_type = w.Dropdown(options=["fixed_season", "gamma_ac"], value="fixed_season", description="Model:", style=_LABEL_STYLE)
    r_experiment_root = w.Text(value=SUGGESTED_EXPERIMENT_ROOT,
                                description="Experiment root:", placeholder="a directory you own -- replace the suggestion above",
                                style=_LABEL_STYLE, layout=_wide())
    r_experiment_name = w.Text(description="Experiment name:", placeholder="directory name under experiment_root -- also names the saved config file", style=_LABEL_STYLE, layout=_wide())
    r_season = w.Dropdown(options=["DJF", "JJA", "MAM", "SON"], value=FIXED_CONTROL["season"], description="Season:", style=_LABEL_STYLE)
    # r_heating_name lives inside heating_gen_box/ss_gen_box below (it's part of
    # "what am I generating", not a top-level run setting) -- defined here since
    # many other widgets below reference it, but never placed at the top level.
    # Its value is entirely auto-generated from whichever Heating/Shape-scale
    # source is selected below (see on_heating_gen_mode_change/on_ss_gen_mode_change
    # and the per-file-field observers further down) -- still editable afterward,
    # same as every other auto-filled field in this notebook.
    r_heating_name = w.Text(value=FIXED_CONTROL["heating_name"], description="Heating name:", style=_LABEL_STYLE, layout=_wide())
    r_start_year = w.IntText(value=FIXED_CONTROL["start_year"], description="Start yr:", style=_LABEL_STYLE)
    r_end_year = w.IntText(value=FIXED_CONTROL["end_year"], description="End yr:", style=_LABEL_STYLE)
    r_preprocess_path = w.Text(value=FIXED_CONTROL["preprocess_path"],
                                description="Preprocess dir:", placeholder="path to base climatology .pt files", style=_LABEL_STYLE, layout=_wide())
    r_run_length_days = w.IntText(value=30, description="Run length (days):", style=_LABEL_STYLE)
    r_cold_start = w.Checkbox(value=True, description="Cold start", style=_LABEL_STYLE)
    r_toffset = w.IntText(value=0, description="Restart offset (days):", style=_LABEL_STYLE)
    r_shape_file = w.Text(description="Shape file:", placeholder="(gamma_ac) blank = control default shapeAC.pt", style=_LABEL_STYLE, layout=_wide())
    r_scale_file = w.Text(description="Scale file:", placeholder="(gamma_ac) blank = control default scaleAC.pt", style=_LABEL_STYLE, layout=_wide())
    r_control_experiment = w.Text(description="Control exp:", placeholder="(optional) name of a control run to diff against", style=_LABEL_STYLE, layout=_wide())
    r_spinup_days = w.IntText(value=60, description="Spinup days:", style=_LABEL_STYLE)
    # Drives both step 3 (postprocess -- interpolate to pressure levels) and
    # step 4 (plot) with the same variable list. Previously two separate fields
    # (Postprocess vars / Plot vars) that could disagree -- e.g. listing a var
    # in Plot vars that wasn't in Postprocess vars would fail at step 4 since
    # the file it needs was never written. One field removes that failure mode;
    # defaults to all four available variables.
    r_plot_vars = w.SelectMultiple(options=["uvel", "vvel", "geo", "temp"], value=["uvel", "vvel", "geo", "temp"],
                                    description="Vars (postprocess &amp; plot):", style=_LABEL_STYLE)

    def experiment_dir():
        return os.path.join(r_experiment_root.value, r_experiment_name.value)


    # --- Restart offset auto-detection ---
    # Both model runners write every output chunk as {var}_{start}_{end}.nc with
    # dates anchored to this same fixed epoch (scripts/02_run_model.py:146,
    # Gamma_AC_Model/RunModel.Gamma.py:106 -- both pd.date_range(start="1950-01-01", ...)),
    # independent of the experiment's own start_year/end_year. toffset is just
    # "days since this epoch" (02_run_model.py's own docstring: "toffset=<days
    # already run>"), so it can be derived directly from what's already on disk
    # instead of hand-computed.
    _MODEL_EPOCH = datetime.date(1950, 1, 1)
    _UVEL_CHUNK_RE = re.compile(r"^uvel_\d{4}-\d{2}-\d{2}_(\d{4}-\d{2}-\d{2})\.nc$")


    def _days_already_run(exp_dir):
        end_dates = []
        for f in glob.glob(os.path.join(exp_dir, "uvel_*.nc")):
            m = _UVEL_CHUNK_RE.match(os.path.basename(f))
            if m:
                end_dates.append(datetime.date.fromisoformat(m.group(1)))
        if not end_dates:
            return None
        return (max(end_dates) - _MODEL_EPOCH).days + 1


    r_toffset_status = w.HTML()


    def _update_toffset_status(*_):
        if r_cold_start.value:
            r_toffset.value = 0
            r_toffset_status.value = "<i>Cold start — Restart offset is 0.</i>"
            return
        exp_dir = experiment_dir()
        days = _days_already_run(exp_dir)
        if days is None:
            r_toffset_status.value = (f"<span style='color:#b30000'>No existing output found in "
                                       f"<code>{exp_dir}</code>.</span> Set Restart offset manually, "
                                       "or check Cold start to begin fresh.")
        else:
            r_toffset_status.value = (f"<span style='color:green'>Detected {days} day(s) already run in "
                                       f"<code>{exp_dir}</code>.</span> Restart offset set to match "
                                       "(edit it yourself if you want something else).")
            r_toffset.value = days


    r_cold_start.observe(_update_toffset_status, names="value")
    r_experiment_root.observe(_update_toffset_status, names="value")
    r_experiment_name.observe(_update_toffset_status, names="value")
    _update_toffset_status()

    # --- Heating generator (fixed_season) ---
    # Two levels: which broad path (existing file vs generate new), then a
    # sub-choice within that path. This keeps the (a)/(b)/(c) explanatory
    # content (below) visible as soon as "generate new" is picked, rather than
    # hidden behind also having to pick a specific anomaly source first.
    PATH_EXISTING = "Use an existing heating file"
    PATH_GENERATE = "Generate a new heating file from a precip anomaly"
    r_heating_path_mode = w.Dropdown(options=[PATH_EXISTING, PATH_GENERATE], value=PATH_EXISTING,
                                      description="Heating source:", style=_LABEL_STYLE, layout=_wide())

    EXISTING_MODES = ["Control default (JJA 1979-2023)", "Control default (DJF 1999-2020)", "Custom file"]
    _FIXED_CONTROL_BY_EXISTING_MODE = {
        EXISTING_MODES[0]: FIXED_CONTROL_JJA,
        EXISTING_MODES[1]: FIXED_CONTROL_DJF,
    }
    r_existing_source_mode = w.Dropdown(options=EXISTING_MODES, value=EXISTING_MODES[0],
                                         description="Existing file:", style=_LABEL_STYLE, layout=_wide())

    ANOMALY_SOURCE_MODES = ["CCA", "CESM2", "CMAP", "CMORPH"]
    _HEATING_SOURCE_BY_ANOMALY = {"CCA": "cca", "CESM2": "cesm2", "CMAP": "cmap", "CMORPH": "cmorph"}
    r_anomaly_source_mode = w.Dropdown(options=ANOMALY_SOURCE_MODES, value=ANOMALY_SOURCE_MODES[0],
                                        description="Anomaly source:", style=_LABEL_STYLE, layout=_wide())

    # Only meaningful for cmap/cmorph -- cca/cesm2's real files are anomaly-only
    # products by construction (no raw multi-year archive of either exists in
    # this repo's data paths), so computing "their own climatology" would be
    # both wrong and impossible; the type toggle is hidden and forced back to
    # Precomputed whenever cca/cesm2 is the selected source (see
    # _on_heating_selection_change below).
    ANOMALY_TYPE_MODES = ["Precomputed anomaly", "Compute anomaly from raw precip"]
    r_anomaly_type_mode = w.Dropdown(options=ANOMALY_TYPE_MODES, value=ANOMALY_TYPE_MODES[0],
                                      description="Anomaly type:", style=_LABEL_STYLE, layout=_wide())

    r_heating_reset_note = w.HTML(
        "<i>Heating name (below) auto-fills from whichever source is picked here -- edit it "
        "yourself if you want something else. Either \"Control default\" option also resets "
        "Season / Start-End yr / Preprocess dir (and Advanced &gt; Heating filename override) to "
        "that control's real values.</i>")
    r_existing_file_info = w.HTML(
        "<i>Points directly at a final, already fully-computed heating tensor (.pt) file -- "
        "it's copied in exactly as-is. No climatology, anomaly, or region/year processing "
        "happens here. A \"Control default\" uses one of the instructor's verified files; "
        "\"Custom file\" copies whatever file you point at below.</i>")
    r_heating_file = w.Text(description="Heating file:", placeholder="path to a pre-made heat.ggrid .pt file to copy in -- Heating name above will auto-fill from it", style=_LABEL_STYLE, layout=_wide())

    r_climo_info = w.HTML(
        "<b>(a) Climatology:</b> NCEP reanalysis monthly precipitation "
        "(input_data_path/precip.mon.mean.nc), averaged over Start&#8211;End yr and the "
        "chosen Season above -- this is the background precip climate the anomaly below "
        "gets added to. Controlled by Season / Start yr / End yr above; not separately "
        "configurable here.")
    r_anomaly_info = w.HTML()  # text set by _update_anomaly_info(), depends on Anomaly type below


    def _update_anomaly_info(*_):
        if r_anomaly_type_mode.value == ANOMALY_TYPE_MODES[0]:
            r_anomaly_info.value = (
                "<b>(b) Anomaly:</b> the file/glob below must already <u>be</u> a precipitation "
                "ANOMALY -- its own climatology has already been removed by whatever process "
                "produced it, before it reaches this tool. This tool does <u>not</u> compute "
                "anomalies from raw precipitation in this mode.")
        else:
            group = "month" if r_anomaly_source_mode.value == "CMAP" else "day-of-year"
            r_anomaly_info.value = (
                f"<b>(b) Anomaly:</b> computed here from raw precipitation -- this dataset's own "
                f"climatology (grouped by {group}) is subtracted from it, the same approach "
                "Gamma_AC's own CMAP-based diagnostic heating uses. Not the NCEP-reanalysis "
                "climatology from step (a) above -- that stays separate, added back in afterward.")


    r_anomaly_type_mode.observe(_update_anomaly_info, names="value")
    r_anomaly_source_mode.observe(_update_anomaly_info, names="value")
    _update_anomaly_info()

    r_cesm2_precip_file = w.Text(description="CESM2 anomaly file:",
        placeholder="path to a precomputed CESM2 precip ANOMALY .nc (climatology already removed)",
        style=_LABEL_STYLE, layout=_wide())
    r_cca_precip_file = w.Text(description="CCA anomaly file:",
        placeholder="path to a precomputed CCA precip ANOMALY .nc (climatology already removed)",
        style=_LABEL_STYLE, layout=_wide())
    r_cmap_precip_file = w.Text(
        value="/data/esplab/shared/obs/gridded/atm/precip/monthly/CMAP/precip.mon.mean.nc",
        description="CMAP file:", style=_LABEL_STYLE, layout=_wide())
    r_cmorph_precip_glob = w.Text(
        value="/data/esplab/shared/obs/gridded/atm/precip/daily/CMORPH/CMORPH_V1.0_ADJ_0.25deg-DLY_00Z_*.nc",
        description="CMORPH glob:", style=_LABEL_STYLE, layout=_wide())

    r_anomaly_mods_label = w.HTML(
        "<b>(c) Modifications</b> to the anomaly (both optional; blank = use the anomaly "
        "file as-is, today's behavior):")
    r_anomaly_lat_min = w.Text(description="Region lat min:",
                                placeholder="optional, e.g. 24 -- blank = no region mask",
                                style=_LABEL_STYLE)
    r_anomaly_lat_max = w.Text(description="Region lat max:",
                                placeholder="optional, e.g. 36", style=_LABEL_STYLE)
    r_anomaly_lon_min = w.Text(description="Region lon min (0-360):",
                                placeholder="optional, e.g. 269", style=_LABEL_STYLE)
    r_anomaly_lon_max = w.Text(description="Region lon max (0-360):",
                                placeholder="optional, e.g. 283", style=_LABEL_STYLE)
    r_anomaly_years = w.Text(
        description="Composite years:",
        placeholder="optional, e.g. 1998 2015 2023 (commas or spaces) -- blank = use the "
                    "anomaly file's Season/Start-End yr as loaded (today's behavior)",
        style=_LABEL_STYLE, layout=_wide())

    heating_gen_output = w.Output()
    heating_gen_button = w.Button(description="Generate Heating File", button_style="warning")


    def _derive_heating_name_from_file(path):
        # Strip extension and the heat.ggrid_/heat_/.ggrid naming-convention
        # noise seen in real files (e.g. heat_DJF_1999-2020_ALL.ggrid.pt ->
        # DJF_1999-2020_ALL) so the auto-filled label is clean, not a literal
        # dump of the source filename.
        base = os.path.splitext(os.path.basename(path))[0]
        for prefix in ("heat.ggrid_", "heat_"):
            if base.startswith(prefix):
                base = base[len(prefix):]
                break
        if base.endswith(".ggrid"):
            base = base[:-len(".ggrid")]
        return base


    def _make_existing_name_deriver():
        # Custom-file field re-derives Heating name when it changes, but only
        # while it's actually the active existing-file sub-choice.
        def handler(change):
            if (r_heating_path_mode.value == PATH_EXISTING
                    and r_existing_source_mode.value == EXISTING_MODES[2] and change["new"]):
                r_heating_name.value = _derive_heating_name_from_file(change["new"])
        return handler


    def _make_anomaly_name_deriver(expected_source_mode):
        # Same idea, one factory shared by the three anomaly precip-file fields:
        # whichever one belongs to the currently-selected anomaly source
        # re-derives Heating name when it changes.
        def handler(change):
            if (r_heating_path_mode.value == PATH_GENERATE
                    and r_anomaly_source_mode.value == expected_source_mode and change["new"]):
                r_heating_name.value = _derive_heating_name_from_file(change["new"])
        return handler


    r_heating_file.observe(_make_existing_name_deriver(), names="value")
    r_cca_precip_file.observe(_make_anomaly_name_deriver("CCA"), names="value")
    r_cesm2_precip_file.observe(_make_anomaly_name_deriver("CESM2"), names="value")
    r_cmap_precip_file.observe(_make_anomaly_name_deriver("CMAP"), names="value")
    r_cmorph_precip_glob.observe(_make_anomaly_name_deriver("CMORPH"), names="value")

    heating_status = w.HTML()


    def _heating_target_path():
        # Mirrors 02_run_model.py's own heating_file resolution -- some real
        # experiments (e.g. the DJF control) point at an existing file whose name
        # predates the heat.ggrid_{heating_name}.pt convention.
        filename = a_heating_file_override.value or f"heat.ggrid_{r_heating_name.value}.pt"
        return os.path.join(r_preprocess_path.value, filename)


    def _update_heating_status(*_):
        p = _heating_target_path()
        if os.path.exists(p):
            heating_status.value = (f"<span style='color:green'>&#10003; Heating file exists:</span> "
                                     f"<code>{p}</code> — Run Pipeline will use this as-is.")
        else:
            heating_status.value = (f"<span style='color:#b30000'>&#10007; Heating file NOT found:</span> "
                                     f"<code>{p}</code> — click Generate above, or Run Pipeline will refuse to start.")


    r_heating_name.observe(_update_heating_status, names="value")
    r_preprocess_path.observe(_update_heating_status, names="value")

    # --- Shape/scale generator (gamma_ac) ---
    SS_GEN_MODES = ["Use control default (shapeAC.pt / scaleAC.pt)", "Fit new: Control period",
                     "Fit new: Composite (e.g. El Nino)", "Generate: No heating (zero)"]
    # Fixed labels for the fit-based modes: there's no source file to derive a
    # name from (they compute new data from a date range, not copy a file), and
    # a stable default matches this notebook's "generate once, then Run Pipeline
    # against exactly that" flow better than baking in dates that would make the
    # name drift every time a date field is tweaked. Still editable afterward.
    _SS_MODE_FALLBACK_NAME = {
        SS_GEN_MODES[1]: "ControlFit", SS_GEN_MODES[2]: "Composite", SS_GEN_MODES[3]: "NoHeating",
    }
    r_ss_gen_mode = w.Dropdown(options=SS_GEN_MODES, value=SS_GEN_MODES[0],
                                description="Shape/scale source:", style=_LABEL_STYLE, layout=_wide())
    r_ss_reset_note = w.HTML(
        "<i>Heating name (below) auto-fills from whichever source is picked here -- edit it "
        "yourself if you want something else. \"Use control default\" also resets Start-End yr / "
        "Preprocess dir to the control values, and clears Shape/Scale file below.</i>")
    ss_precip_glob = w.Text(
        value="/data/esplab/shared/obs/gridded/atm/precip/daily/CMORPH/CMORPH_V1.0_ADJ_0.25deg-DLY_00Z_*.nc",
        description="Precip glob:", style=_LABEL_STYLE, layout=_wide())
    ss_precip_varname = w.Text(value="cmorph", description="Precip var name:", style=_LABEL_STYLE)
    # Default to the full CMORPH archive on disk (confirmed real min/max dates) --
    # this is exactly the range the precomputed regrid cache below was built for,
    # so leaving these at their defaults hits the warm cache and skips the slow
    # reload+regrid entirely. Bump these (and rebuild the cache) if the archive
    # is later extended.
    ss_start_date = w.DatePicker(description="Fit start:", value=datetime.date(1998, 1, 1), style=_LABEL_STYLE)
    ss_end_date = w.DatePicker(description="Fit end:", value=datetime.date(2024, 8, 31), style=_LABEL_STYLE)
    ss_scale_qc_max = w.FloatText(value=300.0, description="Scale QC max:", style=_LABEL_STYLE)
    # Each year becomes a Jul(year)-01 to Jun(year+1)-30 window -- the same
    # water-year convention the real reference notebook used for its actual
    # El Nino composites (e.g. Gamma_AC_Model/reference_notebooks/
    # preprocess.Gamma_heating.ipynb: wyrs=['2002',...]/wyre=['2003',...],
    # Jul-1 to Jun-30), so a peak-DJF ENSO event lands inside one window.
    ss_composite_years = w.Text(
        description="Composite years:",
        placeholder="e.g. 1997 2015 2023 (commas or spaces) -- each becomes a "
                    "Jul(year)\u2013Jun(year+1) window",
        style=_LABEL_STYLE, layout=_wide())
    ss_gen_output = w.Output()
    ss_gen_button = w.Button(description="Generate Shape/Scale Files", button_style="warning")

    ss_status = w.HTML()


    def _ss_target_paths():
        shape_name = r_shape_file.value or "shapeAC.pt"
        scale_name = r_scale_file.value or "scaleAC.pt"
        return (os.path.join(r_preprocess_path.value, shape_name),
                os.path.join(r_preprocess_path.value, scale_name))


    def _update_ss_status(*_):
        shape_path, scale_path = _ss_target_paths()
        parts = []
        for label, p, field in (("Shape", shape_path, r_shape_file), ("Scale", scale_path, r_scale_file)):
            tag = " (control default)" if not field.value else ""
            if os.path.exists(p):
                parts.append(f"<span style='color:green'>&#10003; {label} exists{tag}:</span> <code>{p}</code>")
            else:
                parts.append(f"<span style='color:#b30000'>&#10007; {label} NOT found{tag}:</span> <code>{p}</code>")
        ss_status.value = "<br>".join(parts)


    r_shape_file.observe(_update_ss_status, names="value")
    r_scale_file.observe(_update_ss_status, names="value")
    r_preprocess_path.observe(_update_ss_status, names="value")


    def _parse_composite_years(text_widget):
        v = text_widget.value.strip()
        return [y for y in re.split(r"[,\s]+", v) if y] if v else None


    def _composite_windows_from_years(years):
        # Jul(year)-01 to Jun(year+1)-30 -- see ss_composite_years's definition
        # above for why this specific convention.
        windows = []
        for yr in years:
            yr = int(yr)
            windows.append((f"{yr}-07-01", f"{yr + 1}-06-30"))
        return windows


    # --- Advanced fields (collapsed) ---
    a_zw = w.Dropdown(options=[42, 63, 124], value=63, description="zw:", style=_LABEL_STYLE)
    a_kmax = w.Dropdown(options=[11, 26], value=26, description="kmax:", style=_LABEL_STYLE)
    a_chunk_size_days = w.IntText(value=30, description="Chunk size (days):", style=_LABEL_STYLE)
    a_compute_slp = w.Checkbox(value=False, description="Compute SLP", style=_LABEL_STYLE)
    a_heating_file_override = w.Text(description="Heating filename override:",
                                      placeholder="only if an existing file's name doesn't match heat.ggrid_{heating_name}.pt",
                                      style=_LABEL_STYLE, layout=_wide())
    a_heating_file_override.observe(lambda *_: _update_heating_status(), names="value")
    a_gamma_heating_custom_file = w.Text(description="Diagnostic heat.ggrid file:",
                                          placeholder="rare, gamma_ac only -- NOT used by the running model, purely diagnostic",
                                          style=_LABEL_STYLE, layout=_wide())
    a_precip_cache_dir = w.Text(
        value="/data/esplab/kpegion/projects/AGCM/precip_regrid_cache",
        description="Precip regrid cache dir:",
        placeholder="blank = always reload+regrid the full precip archive from scratch (slow)",
        style=_LABEL_STYLE, layout=_wide())

    advanced_box = w.Accordion(children=[w.VBox([a_zw, a_kmax, a_chunk_size_days, a_compute_slp,
                                                  a_heating_file_override, a_gamma_heating_custom_file,
                                                  a_precip_cache_dir])])
    advanced_box.set_title(0, "Advanced settings (usually leave as default)")
    advanced_box.selected_index = None  # collapsed by default

    config_output = w.Output()
    build_config_button = w.Button(description="Build Config", button_style="info")

    run_output = w.Output()
    run_button = w.Button(description="Run Pipeline (steps 1-4)", button_style="primary")
    r_use_screen = w.Checkbox(
        value=False, description="Run in background (screen)",
        style=_LABEL_STYLE)
    confirm_box = w.VBox([])  # populated with a confirmation button when needed


    def _on_heating_selection_change(*_):
        path = r_heating_path_mode.value
        is_existing = path == PATH_EXISTING
        is_generate_new = path == PATH_GENERATE
        is_custom_existing = is_existing and r_existing_source_mode.value == EXISTING_MODES[2]
        src_mode = r_anomaly_source_mode.value
        # Compute mode only makes sense for cmap/cmorph (raw precip archives) --
        # cca/cesm2's real files are anomaly-only by construction. Force back to
        # Precomputed and hide the toggle for cca/cesm2.
        supports_compute = src_mode in ("CMAP", "CMORPH")
        if not supports_compute and r_anomaly_type_mode.value != ANOMALY_TYPE_MODES[0]:
            r_anomaly_type_mode.value = ANOMALY_TYPE_MODES[0]

        r_existing_source_mode.layout.display = "" if is_existing else "none"
        r_anomaly_source_mode.layout.display = "" if is_generate_new else "none"
        r_anomaly_type_mode.layout.display = "" if (is_generate_new and supports_compute) else "none"

        r_existing_file_info.layout.display = "" if is_existing else "none"
        r_heating_file.layout.display = "" if is_custom_existing else "none"

        # (a)/(b)/(c) become visible as soon as "generate new" is picked, not
        # only once a specific anomaly source is also chosen.
        r_climo_info.layout.display = "" if is_generate_new else "none"
        r_anomaly_info.layout.display = "" if is_generate_new else "none"
        r_cca_precip_file.layout.display = "" if (is_generate_new and src_mode == "CCA") else "none"
        r_cesm2_precip_file.layout.display = "" if (is_generate_new and src_mode == "CESM2") else "none"
        r_cmap_precip_file.layout.display = "" if (is_generate_new and src_mode == "CMAP") else "none"
        r_cmorph_precip_glob.layout.display = "" if (is_generate_new and src_mode == "CMORPH") else "none"
        r_anomaly_mods_label.layout.display = "" if is_generate_new else "none"
        for widget in (r_anomaly_lat_min, r_anomaly_lat_max, r_anomaly_lon_min,
                       r_anomaly_lon_max, r_anomaly_years):
            widget.layout.display = "" if is_generate_new else "none"

        is_control_default = is_existing and r_existing_source_mode.value in (EXISTING_MODES[0], EXISTING_MODES[1])
        heating_gen_button.layout.display = "none" if is_control_default else ""
        if is_control_default:
            control = _FIXED_CONTROL_BY_EXISTING_MODE[r_existing_source_mode.value]
            real_filename = control["heating_file_override"] or f"heat.ggrid_{control['heating_name']}.pt"
            r_heating_name.value = _derive_heating_name_from_file(real_filename)
            r_season.value = control["season"]
            r_start_year.value = control["start_year"]
            r_end_year.value = control["end_year"]
            r_preprocess_path.value = control["preprocess_path"]
            a_heating_file_override.value = control["heating_file_override"]
        elif is_custom_existing:
            r_heating_name.value = (_derive_heating_name_from_file(r_heating_file.value)
                                     if r_heating_file.value else "Custom")
        elif is_generate_new:
            file_field = {"CCA": r_cca_precip_file, "CESM2": r_cesm2_precip_file,
                          "CMAP": r_cmap_precip_file, "CMORPH": r_cmorph_precip_glob}[src_mode]
            r_heating_name.value = (_derive_heating_name_from_file(file_field.value)
                                     if file_field.value else src_mode)
        _update_anomaly_info()


    r_heating_path_mode.observe(_on_heating_selection_change, names="value")
    r_existing_source_mode.observe(_on_heating_selection_change, names="value")
    r_anomaly_source_mode.observe(_on_heating_selection_change, names="value")
    _on_heating_selection_change()


    def on_ss_gen_mode_change(change):
        mode = change["new"]
        is_control_fit = mode == "Fit new: Control period"
        is_composite = mode == "Fit new: Composite (e.g. El Nino)"
        is_default = mode == SS_GEN_MODES[0]
        for widget in (ss_precip_glob, ss_precip_varname, ss_start_date, ss_end_date):
            widget.layout.display = "" if (is_control_fit or is_composite) else "none"
        ss_scale_qc_max.layout.display = "" if is_composite else "none"
        ss_composite_years.layout.display = "" if is_composite else "none"
        ss_gen_button.layout.display = "none" if is_default else ""
        if is_default:
            r_heating_name.value = GAMMA_CONTROL["heating_name"]
            r_start_year.value = GAMMA_CONTROL["start_year"]
            r_end_year.value = GAMMA_CONTROL["end_year"]
            r_preprocess_path.value = GAMMA_CONTROL["preprocess_path"]
            r_shape_file.value = ""
            r_scale_file.value = ""
        else:
            r_heating_name.value = _SS_MODE_FALLBACK_NAME[mode]


    r_ss_gen_mode.observe(on_ss_gen_mode_change, names="value")
    on_ss_gen_mode_change({"new": r_ss_gen_mode.value})


    def on_model_type_change(change):
        is_fixed = change["new"] == "fixed_season"
        r_season.layout.display = "" if is_fixed else "none"
        r_shape_file.layout.display = "none" if is_fixed else ""
        r_scale_file.layout.display = "none" if is_fixed else ""
        heating_gen_box.layout.display = "" if is_fixed else "none"
        ss_gen_box.layout.display = "none" if is_fixed else ""
        if is_fixed:
            r_heating_path_mode.value = PATH_EXISTING
            r_existing_source_mode.value = EXISTING_MODES[0]
            _on_heating_selection_change()
        else:
            r_ss_gen_mode.value = SS_GEN_MODES[0]
            on_ss_gen_mode_change({"new": SS_GEN_MODES[0]})


    r_model_type.observe(on_model_type_change, names="value")


    def _parse_optional_float(text_widget, label):
        v = text_widget.value.strip()
        if not v:
            return None
        try:
            return float(v)
        except ValueError:
            raise ValueError(f"{label} must be a number, got {v!r}.")


    def _parse_anomaly_years(text_widget):
        v = text_widget.value.strip()
        return [y for y in re.split(r"[,\s]+", v) if y] if v else None


    def _add_anomaly_mod_kwargs(kwargs):
        # Shared by on_generate_heating_clicked and build_cfg_dict so the two
        # never drift out of sync on how blank fields map to omitted cfg keys.
        lat_min = _parse_optional_float(r_anomaly_lat_min, "Region lat min")
        lat_max = _parse_optional_float(r_anomaly_lat_max, "Region lat max")
        lon_min = _parse_optional_float(r_anomaly_lon_min, "Region lon min")
        lon_max = _parse_optional_float(r_anomaly_lon_max, "Region lon max")
        if lat_min is not None:
            kwargs["anomaly_lat_min"] = lat_min
        if lat_max is not None:
            kwargs["anomaly_lat_max"] = lat_max
        if lon_min is not None:
            kwargs["anomaly_lon_min"] = lon_min
        if lon_max is not None:
            kwargs["anomaly_lon_max"] = lon_max
        years = _parse_anomaly_years(r_anomaly_years)
        if years:
            kwargs["anomaly_years"] = years


    _heating_click_state = [0.0]

    def on_generate_heating_clicked(b):
        if _is_duplicate_click(_heating_click_state):
            return
        with heating_gen_output:
            heating_gen_output.clear_output()
            print("Generating... this can take a few minutes for cmap/cmorph sources "
                  "(loading the full precip archive) -- no output here yet just means "
                  "it's still working.")
            heating_status.value = "<i>&#8987; Generating heating file...</i>"
            heating_gen_button.disabled = True
            try:
                path = r_heating_path_mode.value
                heating_source = ("custom" if path == PATH_EXISTING
                                   else _HEATING_SOURCE_BY_ANOMALY[r_anomaly_source_mode.value])
                kwargs = dict(
                    model_type="fixed_season",
                    heating_source=heating_source,
                    heating_name=r_heating_name.value,
                    output_dir=r_preprocess_path.value,
                    season=r_season.value,
                    start_year=r_start_year.value,
                    end_year=r_end_year.value,
                    zw=a_zw.value,
                    kmax=a_kmax.value,
                )
                if path == PATH_EXISTING:
                    kwargs["heating_file"] = r_heating_file.value
                else:
                    src_mode = r_anomaly_source_mode.value
                    if src_mode == "CCA":
                        kwargs["cca_precip_file"] = r_cca_precip_file.value
                    elif src_mode == "CESM2":
                        kwargs["cesm2_precip_file"] = r_cesm2_precip_file.value
                    elif src_mode == "CMAP":
                        kwargs["cmap_precip_file"] = r_cmap_precip_file.value
                    else:
                        kwargs["cmorph_precip_glob"] = r_cmorph_precip_glob.value
                    if r_anomaly_type_mode.value == ANOMALY_TYPE_MODES[1]:
                        kwargs["anomaly_type"] = "compute"
                    _add_anomaly_mod_kwargs(kwargs)
                outfile = generate_heating_file(**kwargs)
                print(f"Done. Wrote: {outfile}")
            except Exception as e:
                print(f"ERROR: {e}")
            finally:
                heating_gen_button.disabled = False
                _update_heating_status()


    heating_gen_button.on_click(on_generate_heating_clicked)


    _ss_click_state = [0.0]

    def on_generate_ss_clicked(b):
        if _is_duplicate_click(_ss_click_state):
            return
        with ss_gen_output:
            ss_gen_output.clear_output()
            print("Generating... on a cache miss, fitting reads the full precip archive "
                  "matched by Precip glob (potentially thousands of files for a "
                  "multi-decade range) and can take several minutes -- no output here "
                  "yet just means it's still working. Once cached, any date range or "
                  "composite years is fast.")
            ss_status.value = "<i>&#8987; Generating shape/scale files...</i>"
            ss_gen_button.disabled = True
            try:
                if not r_heating_name.value:
                    raise ValueError("Heating name is required (used as the shape/scale file label).")
                if not r_preprocess_path.value:
                    raise ValueError("Preprocess dir is required (shape/scale files are written there).")

                mode = r_ss_gen_mode.value
                if mode == "Generate: No heating (zero)":
                    zero_shape_scale(r_preprocess_path.value, r_heating_name.value, a_zw.value)
                else:
                    if not (ss_start_date.value and ss_end_date.value):
                        raise ValueError("Fit start/end date required.")
                    windows = None
                    if mode == "Fit new: Composite (e.g. El Nino)":
                        years = _parse_composite_years(ss_composite_years)
                        if not years:
                            raise ValueError("Add at least one composite year, or switch to 'Fit new: Control period'.")
                        windows = _composite_windows_from_years(years)
                        for yr, (w0, w1) in zip(years, windows):
                            print(f"  Composite year {yr} -> {w0} to {w1}")
                    fit_gamma_shape_scale(
                        precip_glob=ss_precip_glob.value,
                        precip_varname=ss_precip_varname.value,
                        date_range=(str(ss_start_date.value), str(ss_end_date.value)),
                        zw=a_zw.value,
                        output_dir=r_preprocess_path.value,
                        name=r_heating_name.value,
                        composite_windows=windows,
                        scale_qc_max=ss_scale_qc_max.value if mode == "Fit new: Composite (e.g. El Nino)" else None,
                        precip_cache_dir=a_precip_cache_dir.value or None,
                    )
                r_shape_file.value = f"shape_{r_heating_name.value}.pt"
                r_scale_file.value = f"scale_{r_heating_name.value}.pt"
                print(f"Done. Wrote shape_{r_heating_name.value}.pt / scale_{r_heating_name.value}.pt "
                      f"to {r_preprocess_path.value}")
                diagnostic_path = os.path.join(r_preprocess_path.value, f"diagnostic_{r_heating_name.value}.png")
                if os.path.exists(diagnostic_path):
                    display(Image(filename=diagnostic_path))
            except Exception as e:
                print(f"ERROR: {e}")
            finally:
                ss_gen_button.disabled = False
                _update_ss_status()


    ss_gen_button.on_click(on_generate_ss_clicked)

    heating_gen_box = w.VBox(
        [
            w.HTML("<b>Generate Heating File</b> (fixed_season) — the options below all configure this:"),
            r_heating_path_mode, r_heating_reset_note,

            r_existing_source_mode,
            r_existing_file_info,
            r_heating_file,

            r_anomaly_source_mode,
            r_anomaly_type_mode,
            r_climo_info,
            r_anomaly_info,
            r_cca_precip_file, r_cesm2_precip_file, r_cmap_precip_file, r_cmorph_precip_glob,
            r_anomaly_mods_label,
            w.HBox([r_anomaly_lat_min, r_anomaly_lat_max]),
            w.HBox([r_anomaly_lon_min, r_anomaly_lon_max]),
            r_anomaly_years,

            r_heating_name, heating_status,
            heating_gen_button, heating_gen_output,
        ],
        layout=_group_box(),
    )

    ss_gen_box = w.VBox(
        [
            w.HTML("<b>Generate Shape/Scale Files</b> (gamma_ac — what the model actually uses) — "
                   "the options below all configure this:"),
            r_ss_gen_mode, r_ss_reset_note,
            r_heating_name, ss_status,
            ss_precip_glob, ss_precip_varname, ss_start_date, ss_end_date, ss_scale_qc_max,
            ss_composite_years,
            ss_gen_button, ss_gen_output,
        ],
        layout=_group_box(),
    )

    postprocessing_box = w.VBox(
        [
            w.HTML("<b>Post-processing &amp; Plotting</b> — only used after the run finishes, to build figures:"),
            r_spinup_days, r_control_experiment, r_plot_vars,
        ],
        layout=_group_box(),
    )

    on_model_type_change({"new": r_model_type.value})
    _update_heating_status()
    _update_ss_status()


    def build_cfg_dict():
        if not r_heating_name.value:
            raise ValueError("Heating name is required.")
        if not r_preprocess_path.value:
            raise ValueError("Preprocess dir is required.")
        if not r_experiment_root.value:
            raise ValueError("Experiment root is required (use your own directory).")
        if not r_experiment_name.value:
            raise ValueError("Experiment name is required.")

        cfg = {
            "model_type": r_model_type.value,
            "start_year": r_start_year.value,
            "end_year": r_end_year.value,
            "season": r_season.value if r_model_type.value == "fixed_season" else "annual",
            "heating_name": r_heating_name.value,
            "preprocess_path_override": r_preprocess_path.value,
            "experiment_root": r_experiment_root.value,
            "experiment_name": r_experiment_name.value,
            "run_length_days": r_run_length_days.value,
            "cold_start": r_cold_start.value,
            "toffset": r_toffset.value,
            "control_experiment": r_control_experiment.value or None,
            "spinup_days": r_spinup_days.value,
            # Same list drives both step 3's pressure-level interpolation and
            # step 4's plotting -- see r_plot_vars's definition above for why.
            "plot_vars": list(r_plot_vars.value),
            "postprocess_vars": list(r_plot_vars.value),
            "zw": a_zw.value,
            "kmax": a_kmax.value,
            "chunk_size_days": a_chunk_size_days.value,
            "compute_slp": a_compute_slp.value,
        }
        if r_model_type.value == "fixed_season":
            cfg["model_subtype"] = "weakly_prescribed_mean"
            # heating_file is what step 1 (mt_preprocess_heating) reads for
            # heating_source=custom -- the raw file to copy in when generating a
            # new heating file. heating_file_override is a separate, unrelated
            # thing: a step-2-only filename override for when an existing file
            # on disk doesn't match the heat.ggrid_{heating_name}.pt formula.
            if r_heating_path_mode.value == PATH_EXISTING:
                cfg["heating_source"] = "custom"
                if r_heating_file.value:
                    cfg["heating_file"] = r_heating_file.value
            else:
                cfg["heating_source"] = _HEATING_SOURCE_BY_ANOMALY[r_anomaly_source_mode.value]
                if r_cca_precip_file.value:
                    cfg["cca_precip_file"] = r_cca_precip_file.value
                if r_cesm2_precip_file.value:
                    cfg["cesm2_precip_file"] = r_cesm2_precip_file.value
                if r_cmap_precip_file.value:
                    cfg["cmap_precip_file"] = r_cmap_precip_file.value
                if r_cmorph_precip_glob.value:
                    cfg["cmorph_precip_glob"] = r_cmorph_precip_glob.value
                if r_anomaly_type_mode.value == ANOMALY_TYPE_MODES[1]:
                    cfg["anomaly_type"] = "compute"
                _add_anomaly_mod_kwargs(cfg)
            if a_heating_file_override.value:
                cfg["heating_file_override"] = a_heating_file_override.value
        else:
            if r_shape_file.value:
                cfg["shape_file_override"] = r_shape_file.value
            if r_scale_file.value:
                cfg["scale_file_override"] = r_scale_file.value
            # heat.ggrid_{heating_name}.pt for gamma_ac is a diagnostic artifact
            # only (never loaded by RunModel.Gamma.py) and is never required for
            # preprocess completeness, so this only matters if a full (non
            # --heating-only) preprocess ever has to run from scratch. Keys are
            # named diagnostic_heating_* (not heating_source/heating_file, which
            # mean the REAL heating for fixed_season) so a saved config can't be
            # misread as describing this run's actual heating mechanism -- for
            # gamma_ac that's entirely shape_file_override/scale_file_override.
            if a_gamma_heating_custom_file.value:
                cfg["diagnostic_heating_source"] = "custom"
                cfg["diagnostic_heating_file"] = a_gamma_heating_custom_file.value
            else:
                cfg["diagnostic_heating_source"] = "cmap_default"
        return cfg


    def config_path():
        # Experiment name drives the saved config filename directly, so there's
        # no separate "Save config to" field that can drift out of sync with it.
        # Written into config/experiments/ alongside the curated experiment
        # configs, per that directory's own convention (one YAML per experiment).
        return os.path.join(project_root, "config", "experiments", f"{r_experiment_name.value}.yaml")


    def on_build_config_clicked(b):
        with config_output:
            config_output.clear_output()
            try:
                cfg = build_cfg_dict()
                path = config_path()
                with open(path, "w") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)
                print(f"Wrote config to {path}")
                print(yaml.safe_dump(cfg, sort_keys=False))
            except Exception as e:
                print(f"ERROR: {e}")


    build_config_button.on_click(on_build_config_clicked)


    def _check_inputs_ready():
        # No input file is ever generated as a side effect of Run Pipeline --
        # either the required file(s) already exist, or Run Pipeline refuses to
        # start and says exactly what's missing and what to click.
        problems = []
        if r_model_type.value == "fixed_season":
            p = _heating_target_path()
            if not os.path.exists(p):
                problems.append(f"Heating file not found: {p} -- click 'Generate Heating File' above, "
                                 "or fix Heating name / Preprocess dir.")
        else:
            for label, p in zip(("Shape file", "Scale file"), _ss_target_paths()):
                if not os.path.exists(p):
                    problems.append(f"{label} not found: {p} -- click 'Generate Shape/Scale Files' above, "
                                     "or fix the filename / Preprocess dir.")
        return problems


    def _run_pipeline(force=False):
        # Chaining logic lives in scripts/run_pipeline.py (shared with its own
        # CLI) so there's exactly one implementation of "run these stages in
        # order, stop at the first failure" for both the notebook and the
        # command line. force=True is only passed once the user has clicked
        # the cold-start overwrite confirmation button below -- it's what
        # lets 02_run_model.py's own --force safeguard proceed with deleting
        # existing output, instead of refusing.
        with run_output:
            run_output.clear_output()
            path = config_path()
            if not os.path.exists(path):
                print("ERROR: build the config first.")
                return
            run_pipeline_stages(path, project_root, print_fn=print, force=force)


    def _run_pipeline_screen(force=False):
        # Same stages, same stop-at-first-failure behavior as _run_pipeline,
        # but launched detached via `screen` so a long run (e.g. gamma_ac's
        # real 150-year control) survives kernel/notebook disconnection
        # instead of blocking it for the run's entire duration. Delegates to
        # run_pipeline.py's own --screen handling rather than re-building the
        # screen/shell-chain here.
        with run_output:
            run_output.clear_output()
            path = config_path()
            if not os.path.exists(path):
                print("ERROR: build the config first.")
                return
            cmd = [sys.executable, os.path.join(project_root, "scripts", "run_pipeline.py"),
                   "--config", os.path.abspath(path), "--screen"]
            if force:
                cmd.append("--force")
            result = subprocess.run(cmd, cwd=os.path.join(project_root, "scripts"),
                                     capture_output=True, text=True)
            print(result.stdout)
            if result.returncode != 0:
                print(result.stderr)
                print("ERROR: failed to launch screen session.")


    def on_run_clicked(b):
        with run_output:
            run_output.clear_output()
            problems = _check_inputs_ready()
            if problems:
                print("Cannot run -- required input file(s) missing:")
                for msg in problems:
                    print(" -", msg)
                print("Nothing has been run.")
                return
        run = _run_pipeline_screen if r_use_screen.value else _run_pipeline
        target = experiment_dir()
        if r_cold_start.value and os.path.isdir(target):
            confirm_btn = w.Button(description=f"Confirm: delete and restart {target}", button_style="danger")

            def on_confirm(cb):
                confirm_box.children = ()
                run(force=True)

            confirm_btn.on_click(on_confirm)
            confirm_box.children = (confirm_btn,)
            with run_output:
                print(f"cold_start=True and {target} already exists — click above to confirm overwrite.")
        else:
            run()


    run_button.on_click(on_run_clicked)

    run_panel = w.VBox([
        w.HTML("<b>Configure &amp; Run Experiment</b>"),
        r_model_type, r_experiment_root, r_experiment_name,
        r_season, r_start_year, r_end_year, r_preprocess_path,
        r_run_length_days, r_cold_start, r_toffset, r_toffset_status,
        heating_gen_box, ss_gen_box, r_shape_file, r_scale_file,
        postprocessing_box,
        advanced_box,
        build_config_button, config_output,
        r_use_screen, run_button, confirm_box, run_output,
    ])

    display(run_panel)
    return locals()
