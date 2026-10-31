# Commons phase diagram: research code

Associated manuscript: *Scarcity-contingent allocation with re-inclusion sustains a regenerating commons across models of behaviour*.
Authors: Tianyu Shi, Hanqing Qian and Chengyi Tu.

Repository: https://github.com/Q-Bigdata/commons-phase-diagram.
Archived version **v1.0.0**: [10.5281/zenodo.23836115](https://doi.org/10.5281/zenodo.23836115), corresponding to source commit `a420e9ed452f95431d6c2934f791229426afdfbf` and [GitHub release v1.0.0](https://github.com/Q-Bigdata/commons-phase-diagram/releases/tag/v1.0.0).
Citation metadata added on the main branch after archival does not change that fixed code version.

## Contents and installation

`code/` contains the simulator, simulation driver, analysis, plot generation and smoke checks. `data/processed/` contains supplied simulation outputs and aggregate analysis results used by the figures. `data/human/` and `data/field/` contain aggregate analysis outputs. See [DATA_SOURCES.md](DATA_SOURCES.md) for provenance and exclusions.

Use a virtual environment and `python -m pip install -r requirements-tested.txt`. The local validation used Python 3.13 with the versions recorded in that file. The manuscript's production environment was reported as Python 3.11, statsmodels 0.14.5 and Numba 0.65.1; that production environment and cross-version bitwise reproducibility have not been independently reproduced.

From this repository root:

```bash
python code/smoke.py
python code/make_figures.py data/processed figures
```

The figure command produces all six main and six Extended Data figures, each as PDF and PNG. Failure to generate any figure now yields a nonzero process status.

## Reproduction boundaries

The main search uses 740 rules in 112 environments. The current robustness entry point uses 1,100 rules in 106 additional environments. The per-rule list and run logs for the archived robustness summaries were not supplied, so their historical rule count cannot be recovered from these aggregate tables.

In `sim.py`, Gini, the worst-to-mean payoff ratio and the probability of ever being excluded are computed for agents who are not fixed free riders (`fr < 0.5`). The human-data Gini in `human.py` uses all four players; these are distinct populations.

The local audit regenerated 12 figures from supplied tables, compared them visually with the manuscript figures, regenerated all 146 quoted-number macros unchanged, and ran small simulator checks. This is not an independent rerun of the full parameter search, 90 neural-planner trainings, human bootstrap analyses or field permutation analyses.

To rerun all experiments, obtain the original public human per-round data and Baggio supplementary input, prepare the inputs expected by `human.py` and `field.py`, and follow the driver and processing stages. The current package does not contain production raw `out/` outputs, saved planner-weight `.npy` files or a verified complete raw-data acquisition workflow. Retraining would be needed to regenerate absent network weights. Original human records, source supplement text and reconstructed source case records are excluded; obtain them separately under their original terms.

```bash
python code/human.py /path/to/human_rounds.csv.gz out_human
python code/field.py /path/to/S1_extracted_text.txt /path/to/h4_class_sims.csv out_field
```

Experiment stages are listed at the end of `code/run_rev.py`. They use `REV_OUT` (default `out`) and `REV_HUMAN` (default `out_human`). Inspect the driver before launching computationally expensive stages. `code/ms_tables.py` contains table-generation routines. The core simulator is unchanged; packaging fixes make plotting inputs discoverable, return nonzero on figure failures, and remove an ineffective `or True` from an integer-payoff assertion.

## Licensing and citation

MIT applies to the original code. It does not relicense third-party source data. The included CSV/JSON files are simulation outputs and aggregate results computed for this study; consult the original sources for their data-use terms. Author and software metadata are in `CITATION.cff`.

Shi, T., Qian, H. & Tu, C. (2026). *Commons phase diagram: research code* (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.23836115.
