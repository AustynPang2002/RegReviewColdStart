# RegReviewColdStart

Research code accompanying **Retrospective drug registration review prediction with FT-based competing risks: multi-horizon comparisons and name-cold evaluation in China**.

The package collects the final training and review-analysis code used for the revised manuscript. It includes six neural variants across five seeds, original and tuned baselines, paired name-cluster bootstrap evaluation, auxiliary-loss sensitivity, device-crossover comparison, calibration diagnostics and independent source-date reconciliation. The modelling, loss and metric code is preserved; packaging changes concern paths, documentation and plotting entry points. This release contains code and documentation only; no research data or result files are distributed.

This is a retrospective research implementation, not a validated prospective decision system. The inferred observation cutoff, historical predictor availability, outcome-dependent eligibility and device-confounded original sensitivity study remain limitations. A fixed seed does not guarantee identical CPU/CUDA/NPU results.

## Install and test

Run commands from the repository root. Python 3.12 was used in the recorded NPU runs.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-deps ./vendor/rtdl-revisiting-models/package
python -m pytest tests -q
```

The default requirements are compatibility ranges, not a locked reproduction environment. The NPU analysis dependency versions are recorded in `requirements-npu-analysis.txt`. For Ascend NPU, install a matched vendor torch/torch_npu/CANN stack first, then `requirements-npu-analysis.txt` and the bundled RTDL package with `--no-deps`; do not use the default requirements to replace an NPU torch build. The original runs used Ascend 910B2. GPU/NPU execution is not tested by the release's local CPU checks.

## Data and configuration

No source workbook, downloaded record, application-level prediction, checkpoint, fitted pickle, server address or credential is included. No aggregate diagnostics or reference-result files are included either. The author-provided data sources are listed in [DATA_SOURCES.md](DATA_SOURCES.md). Obtain inputs directly through authorized access to the applicable sources. Consequently the package is not a self-contained reproduction of the original scores.

Copy `config.example.json` to `config.json`. Set `archive` to an authorized ZIP containing the source Excel workbook, and `member` to its internal path. Keep the study outcome mappings and split dates when reproducing the reported analysis. Run:

```bash
python audit.py
python prepare.py
```

`audit.py` creates the private raw-record cache; `prepare.py` builds the private cohort and manifest. Input fields, date semantics and restrictions are documented in `docs/DATA.md`. The final approval audit contains intentional dataset-specific checks for the paper's original cohort; it is not a generic validation utility for arbitrary databases.

## Reproduce the final analysis

The following commands are full-cohort experiments, not quick tests:

```bash
python ablation_study.py --device cpu --epochs 15 --batch-size 256 --threads 4 --seeds 42 43 44 45 46 --output results/server01_ablation_5seeds
python baselines.py
python review_revision.py --bootstrap 1000
python round2_diagnostics.py --output results/final_diagnostics
python final_approval_audit.py --project-root . --output results/approval_audit
python plot_calibration.py --input results/final_diagnostics/calibration_bins.csv --output results/calibration_diagnostics.png
```

Choose `--device cuda:0` or `--device npu:0` only in an appropriately configured environment. The historical name `server01_ablation_5seeds` is an internal folder convention, not a server connection. It is retained because downstream analyses expect that path. `baselines.py` creates the original baseline fits; `review_revision.py` adds validation-tuned comparators. Refit results may differ from the archived paper outputs, particularly across hardware or library versions. Do not overwrite archived study directories or mix refitted and archived predictions without documenting the distinction.

For a small execution check after preparing data:

```bash
python train.py --device cpu --epochs 1 --max-per-split 200 --output runs/smoke
```

The smoke run cannot be used as a manuscript result. Sensitivity and crossover workflows, including the original device confounding, are detailed in `docs/EXPERIMENTS.md`.

## Repository map

| Files | Purpose |
|---|---|
| `audit.py`, `prepare.py`, `features.py` | Source audit, eligibility, temporal labels, history and train-fitted preprocessing |
| `model.py`, `review_losses.py`, `train.py`, `device_runtime.py` | FT backbone, text fusion, competing risks, auxiliary losses and execution |
| `ablation_study.py`, `baselines.py`, `review_revision.py` | Six-variant study, baselines, comparator tuning and cluster bootstrap |
| `review_sensitivity.py`, `analyze_review_sensitivity.py`, `analyze_crossover.py` | Auxiliary-loss and device studies |
| `round2_diagnostics.py`, `final_approval_audit.py`, `plot_calibration.py` | Calibration/censoring, source dates and the calibration figure |
| `predict.py` | Inference with a paired trusted model and feature checkpoint |
| `tests/`, `requirements-npu-analysis.txt` | Logic tests with synthetic in-memory fixtures and recorded analysis dependencies |

The architecture image was supplied as an illustration; it is not generated by model code and is not included here. This package does not edit the manuscript or publish anything automatically.


## Citation

See `CITATION.cff`. This is an unpublished manuscript; no DOI, acceptance or journal publication is claimed.
