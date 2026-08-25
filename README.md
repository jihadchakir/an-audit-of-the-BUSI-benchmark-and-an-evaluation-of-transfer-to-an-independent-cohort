# An audit of the BUSI benchmark and an evaluation of transfer to an independent cohort

Code and analysis for the manuscript:

> **Data leakage and cross-dataset generalization in breast ultrasound
> classification: an audit of the BUSI benchmark and an evaluation of transfer
> to an independent cohort.**
> Jihad Chakir and Mohssin Zekriti, Euro-Mediterranean University of Fes (UEMF).

The repository contains the dataset audit, a leakage-controlled evaluation
pipeline, the ablation switches that quantify each source of leakage, the
cross-dataset transfer diagnostics, and the committed run outputs that back every
number reported in the paper.

## What is and is not included

Included: all source code, the small run outputs under `runs/` (JSON/CSV summaries
and per-fold predictions), the list of duplicate and label-conflict groups, and a
script that regenerates the reported tables from those outputs.

Not included: the raw image datasets and the trained model weights. The datasets
are redistributed by their original providers, and the weights are large and are
not needed to verify the reported numbers.

- **BUSI**: Al-Dhabyani et al., *Dataset of breast ultrasound images*, Data in
  Brief 28:104863 (2020).
- **BUS-BRA**: Gómez-Flores et al., *BUS-BRA: A breast ultrasound dataset for
  assessing computer-aided diagnosis systems*, Medical Physics 51(4):3110–3123
  (2024).

## Quick start: regenerate the reported tables

```bash
pip install -r requirements.txt
python reproduce_paper_numbers.py
```

This reads the committed per-fold predictions and external reports and prints the
two-class argmax operating point (Table 1) and the per-device vendor
stratification, both as mean ± SD over the five outer folds. Expected values:

```
Table 1, two-class (plain argmax):
  Balanced accuracy      0.817 ± 0.036
  Recall (sens), malig   74.5% ± 6.4
  Specificity, malig     89.0% ± 2.8
  PPV, malig             76.5% ± 5.1

Vendor stratification:
  GE Logiq 5 @10-12MHz   0.607 ± 0.018
  GE Logiq 7 @10-14MHz   0.617 ± 0.052
  Toshiba Aplio 300      0.701 ± 0.033
  U-Systems              0.801 ± 0.098
  All devices            0.607 ± 0.031
```

## Running the full pipeline

Rerunning training and evaluation from the raw data requires the datasets and a
GPU. The main entry points are:

| Script | Purpose |
| --- | --- |
| `audit_dataset.py` | Perceptual-hash audit: exact/near-duplicates, label conflicts, mask facts. |
| `run_cv.py` | Group-aware nested cross-validation under the leakage-controlled protocol; also runs the ablations. |
| `diagnose_mask_leak.py` | Mask-in-input and ROI-crop leakage, including the architecture-free logistic-regression probe. |
| `diagnose_transfer.py` | Cross-dataset transfer to BUS-BRA and the preprocessing/vendor decomposition. |
| `external_eval.py` | Applies a frozen BUSI-trained encoder to the external cohort. |
| `reproduce_paper_numbers.py` | Regenerates the reported tables from committed outputs (no GPU needed). |

Configuration lives in `config.py`; the encoder, heads, losses, splitting, and
decision logic are in `model.py`, `heads.py`, `losses.py`, `splits.py`, and
`decision.py`.

## Tests

```bash
python run_tests_no_pytest.py
```

The suite checks the group-aware splitting is leakage-free, that the feature
widths match the manuscript (270 for two-class, 275 for three-class), that
thresholds chosen on validation are applied verbatim to the test fold, and that
re-enabling a defect inflates the measured performance as reported.

## Repository layout

```
audit_dataset.py, review_conflicts.py     dataset audit and conflict handling
config.py, data.py, splits.py             configuration, data pipeline, splitting
model.py, heads.py, losses.py             encoder, classifier heads, training loss
train_encoder.py, run_cv.py               training and nested cross-validation
decision.py, metrics.py                   operating points and evaluation metrics
diagnose_mask_leak.py, diagnose_transfer.py   leakage and transfer diagnostics
external_eval.py, inspect_external.py     external-cohort evaluation
reproduce_paper_numbers.py                regenerate reported tables from runs/
run_tests_no_pytest.py, test_pipeline.py  regression tests
runs/                                     committed audit and evaluation outputs
CHANGES.md                                defects found in the original pipeline and their fixes
```

`CHANGES.md` documents, cell by cell, the defects found in the original notebooks
and the fix applied to each; it is the reproducibility record behind the paper's
central claims.

## License

Released under the MIT License. See `LICENSE`.

## Citation

A citation will be added here once the paper is published. Until then, please cite
the manuscript by title and authors as listed above.
