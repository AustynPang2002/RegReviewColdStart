# Final experiment sequence and limitations

## Primary study

`ablation_study.py` defines full, ft_base, no_text, no_history, no_decay and no_multitask, each with seeds 42-46 (30 fits). The reported budget is 15 epochs, batch size 256, four CPU threads and early stopping after five non-improving validation-NLL epochs. Checkpoints are selected on validation NLL. Never select a preferred seed using the test set.

The feature class retains a compatibility branch for older serialized preprocessors. Newly fitted preprocessors use training-median imputation; historical objects retain their recorded behaviour. Thus rerunning current source is not a promise to regenerate every historical object byte-for-byte. Source hashes and data fingerprints are recorded by the experiment launchers.

## Auxiliary-loss sensitivity

The historical 2x2 study used reference and weight_normalized on one NPU card, supported_consistency and both_changes on another. This confounds consistency policy with device context; the results remain descriptive. To generate two worker outputs (run separately, selecting the desired physical device externally):

```bash
python review_sensitivity.py --device npu:0 --variants reference weight_normalized --output runs/worker6
python review_sensitivity.py --device npu:0 --variants supported_consistency both_changes --output runs/worker7
```

To use the original `analyze_review_sensitivity.py`, collect all 20 variant_seed directories under `revision_20260925/sensitivity_5seeds/`. Copy each worker's status.json and study_plan.json to that directory as worker6_status.json, worker6_study_plan.json, worker7_status.json and worker7_study_plan.json. Then run:

```bash
python analyze_review_sensitivity.py --bootstrap 1000
```

The analyzer requires completed original-size studies and identical source/data fingerprints. If designing a new study, cross every policy and seed on each device to address confounding; do not label such a study as the historical experiment.

## Matched-seed crossover

Run these separately on the two selected device contexts, preserving identical source, configuration and input data:

```bash
python review_sensitivity.py --device npu:0 --variants reference supported_consistency --seeds 42 --output results/crossover/card6
python review_sensitivity.py --device npu:0 --variants reference supported_consistency --seeds 42 --output results/crossover/card7
python analyze_crossover.py --input results/crossover
```

Selecting the same physical device twice does not reproduce a cross-device comparison. Physical device allocation is external to these commands. The historical crossover comprised only four fits and cannot establish device equivalence across seeds or remove confounding in the earlier study.

## Local diagnostics

No aggregate results are distributed in this code-only release. Generate calibration diagnostics locally, then pass the resulting calibration_bins.csv to plot_calibration.py using --input. The source-date audit retains paper-specific assertions as executable validation logic, not as a distributed dataset.

Bootstrap intervals condition on fitted models and the training censoring estimator and are exploratory without multiplicity correction. Calibration fits are diagnostic only; no test-fitted recalibration is substituted into primary scores.
