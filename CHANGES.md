# What was wrong, and what the rebuild does instead

Every entry cites the notebook and cell it came from. Severity is judged by
whether it changes the reported numbers or only the engineering.

Legend: **[fatal]** invalidates a reported number. **[serious]** biases a number.
**[fix]** correctness or reproducibility, no direct effect on claims.

---

## 1. [fatal] The ground-truth mask was part of the model input

`SNN2.ipynb` cell 5, both ensemble notebooks cell 4:

```python
def apply_mask(self, img, mask):
    mask_3ch = np.stack([mask]*3, axis=-1) / 255.0
    return img * mask_3ch + img * 0.3 * (1 - mask_3ch)
```

Two separate problems, either one sufficient to sink the paper.

**(a) The mask is an expert annotation.** It is the segmentation the model is
supposed to make unnecessary. Feeding it in at inference means the described
system cannot exist: to classify a new scan you would first need a radiologist
to outline the lesion. This objection needs no effect size and no experiment.

**(b) It encodes the label.** Measured on this dataset, not asserted:

> **All 780 BUSI images ship a mask. Exactly 133 of those masks are entirely
> black, and those 133 are exactly the 133 `normal` cases.** The correspondence
> `mask is empty` <=> `class is normal` is 1:1, with no exceptions.

Substitute `mask = 0` into the formula and it collapses to `out = 0.3 * img`, a
uniform darkening of the whole frame. Lesion cases instead keep full intensity
inside the expert's outline. The two classes were preprocessed by **different
functions, selected by the label**.

How much does that leak? `diagnose_mask_leak.py` answers empirically, with a
deliberately trivial probe: logistic regression on 20 global intensity and
gradient statistics, no spatial reasoning, group-aware CV. Measured on BUSI:

| trivial probe, 3-class     | OLD (mask) | NEW (no mask) | chance |
|----------------------------|-----------:|--------------:|-------:|
| balanced accuracy          | **82.31%** |        58.89% | 33.33% |
| recall, normal             |**100.00%** |        60.15% |        |
| recall, benign             |     70.25% |        45.08% |        |
| recall, malignant          |     76.67% |        71.43% |        |

**The `normal` class is recovered at 100% recall, 133/133, by a model that
cannot read an ultrasound.** It has no spatial reasoning whatsoever. It is not
recognising healthy tissue; it is detecting whether an annotation existed. Any
`normal` per-class accuracy in the submitted paper measured the annotation
protocol.

The gap of **+23.42 balanced-accuracy points** is a *lower bound* on the
artefact, because a CNN additionally sees the mask boundary, a hard 3.33x
intensity step that draws the expert's lesion contour into the input as a
visible line. The probe is blind to that channel.

Note the per-class pattern: normal +40 points, benign +25, malignant +5. The
normal gain is the empty-mask identity. The benign gain is consistent with a
second channel, the mask's **area and shape leaking lesion morphology**, size,
roundness and margin regularity being exactly the features that separate benign
from malignant. The 2-class ablation settles this.

**Two diagnostics that were tried and rejected. Do not repeat them:**

1. *Cohen's d of mean brightness*, in `audit_dataset.py`, returns **0.68** and
   badly understates the artefact. Lesion masks cover a small fraction of the
   frame, so averaging dilutes the signal, and the between-image variance in
   scanner gain swamps what is left. The field is retained in the JSON with a
   warning attached; ignore it.
2. *The identity `max(out) <= 0.3`* is true for an all-black mask, since
   `out = 0.3*img` and `img <= 1`. But the converse fails: `max(out) > 0.3`
   requires a pixel **inside** the lesion brighter than 0.3, and breast lesions
   are typically **hypoechoic**, darker than surrounding tissue. The rule catches
   every normal and flags most lesions too. Perfect recall, useless precision.

The lesson generalises: picking a statistic and reasoning that it must carry the
leak is guesswork. Measure it with a probe that cannot do the real task, and let
the ablation settle the magnitude.

**Fix:** `data.py` never touches the mask. Input is the raw grayscale image,
resized, ImageNet-normalised. `use_gt_mask_oracle=True` reproduces the old
behaviour for one ablation row. Label that row as an oracle, never as a result.

**Worth testing in the ablation:** dimming the background to 30% brings bright
tissue *down toward* the dark hypoechoic lesion. The old preprocessing may have
partly destroyed lesion contrast rather than helping it, in which case the mask
was simultaneously leaking the label and degrading the real signal.

---

## 2. [fatal] The threshold was fitted to the set it was reported on

`Complete_Conservative_Ensemble.ipynb` cell 16:

```python
thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
for threshold in thresholds:
    acc, cm, _ = ensemble.evaluate(X_val_paths, y_val, loader)   # the reported set
perfect = [r for r in results if r['malignant_recall'] == 1.0]
best = max(perfect, key=lambda x: x['accuracy'])
```

and then cell 20 went further:

```python
recommended_threshold = missed_case_info['max_conf'] + 0.001
```

The threshold was placed relative to the confidence of the specific malignant
case that the evaluation set contained. That is a parameter fitted to the test
labels. `tests/test_pipeline.py::test_tuning_threshold_on_the_test_set_inflates_sensitivity`
runs that procedure on a classifier with **zero signal** (Dirichlet noise) and
gets 100% malignant sensitivity, every repetition. A number a random classifier
also achieves carries no information about your model.

**Fix:** `decision.py`. `fit_decision(val_proba, val_y, ...)` returns a frozen
`Decision` dataclass; `apply_decision(test_proba, decision)` applies it once.
The `Decision` is serialised into each fold directory, so a reviewer can verify
that the tau applied to test is the tau chosen on validation. The dataclass is
`frozen=True` so it cannot be mutated after selection.

---

## 3. [fatal] The CV ensemble notebook trained and tested on overlapping data

`Complete_Conservative_Ensemble_cross_validation.ipynb` cell 6:

```python
MODEL_PATH = "cv_results/model_fold_1.h5"        # trained on fold 1's train split
...
X_train_paths, X_val_paths, y_train, y_val = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels)
```

The fold-1 encoder was trained on the fold-1 partition from `StratifiedKFold`.
The evaluation here uses a completely different 80/20 partition. The two have no
reason to align, so a large share of the images in `X_val_paths` are images the
encoder was trained on. This is not subtle leakage, it is evaluating on training
data.

**Fix:** `run_cv.py` carries one split object per fold from creation through
training, head fitting, threshold selection and evaluation. Splits are never
re-derived. `splits.assert_no_leakage` runs on every fold and raises (not warns)
if any path or duplicate-group appears in two partitions.

---

## 4. [fatal] Splits were not reproducible across notebooks

`get_image_paths` iterates `os.listdir(class_path)`, whose order is filesystem
dependent and not sorted. `train_test_split(..., random_state=42)` then permutes
that order. Two notebooks running the same line on different machines, or after
the files were copied, get different splits. The comment in the ensemble
notebook, `# Split data (MUST match training split!)`, states an assumption the
code cannot honour.

**Fix:** `audit_dataset.scan_busi` sorts everything and writes an explicit index
CSV. Splits are computed once, saved to `splits.csv`, and passed by index.

---

## 5. [fatal] BUSI's duplicate images were split across train and test

BUSI contains duplicated and near-duplicated images, and even some non-breast
scans; this is documented (Aumente-Maestro, Diez & Remeseiro, *Computer Methods
and Programs in Biomedicine* 260:108540, 2025, which ships a curated BUSI, and
the duplicate discrepancies are noted by others reusing the set). Random
per-image splitting puts one copy in train and another in test, so the model
gets credit for recall, not generalisation.

BUSI also has **no patient identifiers**, so true patient-wise splitting is
impossible. That is a limitation to state explicitly, not to paper over. For
comparison, the BUS benchmark maintained by MIDA refuses results based on random
splitting and requires case-wise partitions.

**Fix:** `audit_dataset.py` detects exact pixel duplicates (md5 of decoded
pixels) and near-duplicates (DCT perceptual hash, Hamming <= 6), builds
connected components, and assigns a `group` id. `splits.py` uses
`StratifiedGroupKFold` so no group straddles a partition.
`duplicate_policy="drop"` gives the stricter one-image-per-group variant; report
both. `tests/test_pipeline.py::test_random_split_inflates_accuracy_versus_group_split`
measures the effect on synthetic data with the same structure: **37.8 points** of
inflation for a 1-NN classifier.

Measured on BUSI:

| audit result | value |
|---|---:|
| images | 780 |
| duplicate groups | 608 |
| images sitting in a multi-image group | **312 (40%)** |
| near-duplicate pairs | 207 |
| exact pixel-duplicate pairs | **1** |
| largest duplicate group | 4 |
| near-duplicate pairs with **conflicting labels** | 10 |
| images with more than one mask file | 17 |

Two things to note. **40% of the dataset sits in a duplicate group**, so random
per-image splitting leaked, and deduplicating to one image per group would take
BUSI from 780 to 608, a 22% reduction. And only **one** pair is an exact pixel
duplicate: the rest are re-encoded or re-cropped near-copies, invisible to a
checksum. A dedup pass built on md5 would have declared BUSI clean and found
1 of the 207.

The **10 pairs with conflicting labels** need visual review before the results
are written up. Either the pHash threshold of 6 is admitting false positives, in
which case tighten it to 4, or BUSI contains near-identical images with
different labels, which is a labelling error and reportable on its own.

---

## 6. [serious] Model selection used a metric that had gone flat

`SNN2.ipynb` cells 7, 11, 13. Triplets were drawn uniformly at random:

```python
anchor_class = np.random.choice(self.class_names)
anchor_idx, positive_idx = np.random.choice(anchor_indices, 2, replace=False)
negative_class = np.random.choice(negative_classes)
```

With three classes and a few hundred images, a random triplet is easy. After a
few epochs the margin is satisfied for nearly all of them and the loss sits near
zero. Meanwhile `val_loss` was computed on **200 freshly drawn random triplets
every epoch**, so it was both saturated and noisy. Everything downstream keyed
off it: `EarlyStopping(monitor='val_loss')`, `ModelCheckpoint(save_best_only)`,
and `best_fold = argmin(val_loss)`.

This is the mechanism behind the observation you had earlier, that a fold with
lower accuracy but "better embedding quality (lower val loss)" won: the val loss
was not measuring embedding quality, it was measuring which random draw of
triplets happened to be easy.

**Fix:** `losses.BatchHardTripletLoss` + `data.PKSampler` (P classes x K images
per batch) keep the gradient alive, and `losses.batch_hard_stats` logs
`frac_active_triplets` so a dead loss is visible. Model selection uses
`train_encoder.prototype_probe`: balanced accuracy of a nearest-prototype probe
on the inner validation split, deterministic given the model, and the quantity
we actually care about.

---

## 7. [serious] Only the best fold was reported

`SNN2.ipynb` cells 13/14 keep only `best_model.h5` (`SAVE_ONLY_BEST_FOLD = True`,
`DELETE_INTERMEDIATE_MODELS = True`), and cell 16 then evaluates that fold on
**its own validation split**, printing:

```
Estimated Full CV Performance:
  Expected Mean: acc-2% - acc+2%
  (Best fold typically 1-3% above mean)
```

Picking the max of five noisy estimates and reporting it, with a hand-waved
correction, is selection bias with a fudge factor. Five folds were trained and
the compute for four was thrown away.

**Fix:** `run_cv.py` trains and evaluates all five outer folds. Every image is
predicted exactly once, by a model that never saw it
(`test_every_image_is_tested_exactly_once`). The headline number is the pooled
out-of-fold estimate with a bootstrap CI; per-fold mean +/- SD is reported
alongside.

---

## 8. [serious] The ensemble mixed incompatible score scales

`ConservativeMalignantEnsemble.predict_class`:

```python
max_malignant_conf = max([conf['malignant'] for conf in confidences_list])
if max_malignant_conf > self.threshold:
    return 'malignant', {...}
```

The three members did not speak the same language. `SiameseClassifier` returned
`1 - dist/max_dist`, a rescaled distance that is **identically 0 for the
farthest class by construction** and has no probabilistic meaning. XGBoost and
Random Forest returned `predict_proba`. Taking a max across those and comparing
it to one threshold is arbitrary: the prototype member's score can never sit on
the same scale as the other two, so the max is dominated by whichever member
happens to be numerically larger, not by whichever is more confident.

This also explains why the two notebooks needed completely different threshold
grids (`[0.10 ... 0.40]` vs `[0.35 ... 0.60]`) for the same rule.

**Fix:** `heads.py`. Every member emits a calibrated posterior over classes.
`PrototypeHead` uses a softmax over negative distances with a temperature fit by
minimising NLL on leave-one-out training distances. `MetaHead` wraps XGBoost/RF
in `CalibratedClassifierCV` with internal K-fold on the training set (so
validation stays free for the operating point). The ensemble is a soft vote.
Output is a probability, so it can be calibrated, thresholded and scored with a
proper scoring rule (Brier, ECE), all of which are now reported.

---

## 9. [serious] Prototype features were self-referential at fit time

`EnhancedSiameseClassifier.train_meta_classifier` built features for each
training embedding using prototypes that were means over that same training set.
Every point pulled its own prototype towards itself, so the distances XGBoost
and RF were fit on were systematically smaller than the distances they would see
at test time. The meta-classifier learned a threshold on a quantity whose
distribution shifts between fit and inference.

**Fix:** `heads.loo_prototypes` computes exact leave-one-out prototypes,
`mean_{-i} = (sum_c - E_i) / (n_c - 1)`, for training rows; shared prototypes
are used at inference. Verified against brute force in
`test_loo_prototypes_match_brute_force`.

---

## 10. [serious] Silent label misalignment on unreadable images

Repeated in every `evaluate`:

```python
batch_images = self.data_loader.load_images_batch(batch_paths)  # drops failures
for img in batch_images:
    predictions.append(pred_idx)
predictions = np.array(predictions)
labels = labels[:len(predictions)]                              # truncates!
```

`load_images_batch` silently drops images that fail to load. The labels are then
truncated from the end rather than at the matching positions, so every label
after the first failure is shifted by one and compared against the wrong image.
If nothing ever failed to load, this is harmless. There is no way to know from
the notebook whether anything failed, because it was never checked. The same
pattern appears in `evaluate_fold`, where a failure defaults to `y_pred.append(0)`,
silently scoring the case as benign.

**Fix:** `data.BUSIDataset.__getitem__` raises on an unreadable image. Nothing is
dropped, nothing is defaulted to benign.

---

## 11. [serious] fp16 embeddings and distances

`SNN2.ipynb` cell 1 set a global `mixed_float16` policy. The final `Dense`, the
`Lambda(l2_normalize)` and therefore every pairwise distance in the triplet loss
and the entire downstream ensemble were computed in fp16 (about 3 decimal
digits). The `EnhancedSiameseClassifier` then computed ratios of those distances,
which amplifies the error.

**Fix:** `model.ResNetFPNEncoder.forward` disables autocast for the embedding
head and normalisation, so embeddings are always fp32. Default AMP is bf16
(see README: better numerics than fp16 and full speed on Blackwell).

---

## 12. [serious] ImageNet weights, non-ImageNet preprocessing

The loader returned `img.astype(np.float32) / 255.0` and fed it straight into a
`ResNet50(weights='imagenet')`. Those weights expect mean/std normalised input.
Loading pretrained weights and then feeding them out-of-distribution input
wastes most of the transfer, and weakens the "pretrained ResNet50" claim.

**Fix:** `data.py` applies ImageNet mean/std. Grayscale is replicated to 3
channels explicitly rather than round-tripping through `COLOR_BGR2RGB` on an
already-gray image.

---

## 13. [fix] The feature count in the paper is wrong

The docstring says *"Extract 278-dimensional features"*. The code builds:

```
256 (embedding) + 9 (3 classes x euclid/cosine/manhattan) + 4 (mean,std,min,max)
+ 3 (ratios) + 3 (top-2 distances and margin) = 275
```

**Fix:** `heads.FeatureSpec` derives the number and `make_features` asserts it.
`run_cv.py` prints the breakdown. Whatever the manuscript says must equal this.
Pinned by `test_feature_width_is_275_not_278`.

---

## 14. [fix] Other things that were quietly broken

- `dummy_labels = np.zeros((len(batch_anchors), 128*3))` while `embedding_size = 256`.
  Harmless because `triplet_loss` ignores `y_true`, but it means the loss
  signature was never actually verified.
- `base_network = trained_model.layers[3]` and `siamese_model.layers[3]`: index
  based lookup into a graph whose layer order is not part of the API.
- Only `_mask.png` was read. BUSI has `_mask_1.png` / `_mask_2.png` for some
  multi-lesion cases. `scan_busi` collects all of them.
- Inference called `.predict()` once per image inside a Python loop, for every
  member, for every threshold in the sweep. `model.extract_embeddings` embeds
  once, in batches, and reuses.
- `START_FROM_FOLD = 4` resume logic merges `fold_results` from a previous JSON
  and carries `best_val_loss` across runs, so the recorded "best fold" can come
  from a different code version than the folds it is compared against.
- The class docstring `"""UPGRADED VERSION ... Expected improvement: 85.9% -> 88-90%"""`.
  Delete this from anything a reviewer sees. Naming a target accuracy before
  running the experiment is exactly the framing that invites a leakage hunt.

---

## 15. [fix] Reporting

The submitted version reported one accuracy per configuration and one confusion
matrix. With ~156 test images and ~42 malignant cases, the 95% CI on malignant
sensitivity spans roughly 10 points. "42/42" without an interval overstates the
evidence.

**Fix:** `metrics.py` reports balanced accuracy, macro F1, per-class
sensitivity/specificity/PPV/NPV, malignant AUROC and AUPRC, Brier, ECE, each
with a stratified percentile bootstrap CI, plus **coverage** next to every
accuracy figure. An abstaining system that is accurate on 60% of cases is not
comparable to one that answers everything, and the paper must not present it as
if it were. `test_coverage_is_reported_with_accuracy` pins this.

---

## What this means for the manuscript

Items 1, 2, 3 and 5 each independently invalidate the headline result. This is
not "the numbers move a little". Expect the honest 3-class balanced accuracy to
land well below the submitted figure, and expect malignant sensitivity at a
fixed operating point to be nowhere near 100%. That is the correct outcome: the
old number was measuring the mask, the threshold search, and the duplicates.

The two measured findings are strong enough to carry a paper by themselves:

* the standard BUSI mask preprocessing recovers the `normal` class at **100%
  recall from a model that cannot read an ultrasound**, because `mask is empty`
  is exactly equivalent to `class is normal` in this dataset;
* **40% of BUSI sits in a near-duplicate group**, and only 1 of 207 duplicate
  pairs is detectable by checksum.

Both are properties of the dataset and of a preprocessing idiom, not of your
model. Anyone using either on BUSI has the same problem.

The rebuilt result will be smaller and defensible, and with external validation
on BUS-BRA it will be worth more to a reviewer at *Biomedical Signal Processing
and Control* than the old one was. The contribution should be reframed around
the honest comparison, and the ablation table (oracle mask on/off, random split
vs group split, threshold on test vs on validation) is itself a useful finding:
it quantifies how much of the published BUSI literature's headroom is artefact.

Also, drop "clinical-grade" from the title. Nothing validated on one 780-image
single-vendor, single-centre dataset is clinical-grade, and a reviewer who sees
that adjective will look harder at everything else.
