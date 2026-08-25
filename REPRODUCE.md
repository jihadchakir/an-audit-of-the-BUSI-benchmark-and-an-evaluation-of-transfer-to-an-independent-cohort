# Reproducing the paper

This repository accompanies the paper *"Data leakage and cross-dataset
generalization in breast ultrasound classification: an audit of the BUSI
benchmark and an evaluation of transfer to an independent cohort."*

It contains the audit, the leakage-controlled cross-validation, the ablation
switches, the transfer diagnostics, and the committed run outputs (the small
JSON/CSV files under `runs/`). The raw image datasets and the trained model
weights are **not** included; the datasets are available from their original
sources (BUSI and BUS-BRA) and the weights are large and not needed to verify
the reported numbers.

## Environment

```bash
pip install -r requirements.txt
```

## Regenerate the reported tables from committed outputs

The two numbers in the paper that are not printed verbatim in a summary file,
the plain-argmax operating point of Table 1 and the per-device vendor
stratification, are regenerated directly from the saved per-fold predictions and
external reports:

```bash
python reproduce_paper_numbers.py
```

Expected output (mean +/- SD over the five outer folds):

```
Table 1, two-class (plain argmax):
  Balanced accuracy      0.817 +/- 0.036
  Recall (sens), malig   74.5% +/- 6.4
  Specificity, malig     89.0% +/- 2.8
  PPV, malig             76.5% +/- 5.1

Vendor stratification:
  GE Logiq 5 @10-12MHz   0.607 +/- 0.018
  GE Logiq 7 @10-14MHz   0.617 +/- 0.052
  Toshiba Aplio 300      0.701 +/- 0.033
  U-Systems              0.801 +/- 0.098
  All devices            0.607 +/- 0.031
```

## Run the full pipeline

To rerun training and evaluation from the raw data (requires the datasets and a
GPU), see `run_cv.py`, `diagnose_mask_leak.py`, `diagnose_transfer.py`, and
`external_eval.py`. The defects found in the original notebooks and the fixes
applied here are documented cell by cell in `CHANGES.md`.

## Tests

```bash
python run_tests_no_pytest.py
```
