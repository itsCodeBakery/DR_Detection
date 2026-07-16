# Project Plan

## Objective

Build and verify an end-to-end PyTorch pipeline for five-class diabetic retinopathy (DR) grading on a pre-resized EyePACS dataset in Kaggle, while controlling GPU memory, preventing validation leakage, and preserving full experiment reproducibility in GitHub.

## Success criteria

A successful project must:

- run from a fresh Kaggle notebook with documented inputs and secrets;
- create a deterministic, class-aware subset when full-data experiments are impractical;
- use leakage-resistant train/validation/test manifests;
- train without out-of-memory failures under a declared Kaggle accelerator;
- report clinically relevant class-wise and ordinal metrics;
- produce publication-quality learning curves, confusion matrices, ROC/PR plots, calibration plots, error galleries, and Grad-CAM figures;
- reproduce every result from a committed configuration and Git commit.

## Stage 0 — Repository and governance

**Deliverables**

- Repository layout, README, license, ignore rules, dependency specification, contribution workflow, and experiment protocol.
- Explicit prohibition on committing data, tokens, checkpoints, and generated bulk artifacts.

**Exit check**

- Repository can be cloned and inspected without any private credential or dataset file.

## Stage 1 — Kaggle runtime and dataset audit

The first executable notebook will perform only lightweight inspection before any training code is introduced.

**Checks**

1. Python, PyTorch, CUDA, GPU model, VRAM, CPU count, RAM, and disk availability.
2. Existence and readability of all supplied paths.
3. CSV columns, row counts, null values, duplicate identifiers, label range, label distribution, and train/test provenance.
4. Image directory size, file-extension distribution, filename-to-label matching, unmatched rows, orphan files, and duplicate stems.
5. A small stratified image sample for dimensions, channels, corruption, black borders, brightness, blur, and orientation.
6. Whether filenames reveal patient/eye pairing such as left/right images.

**Memory policy**

- Never load all pixels into RAM.
- Enumerate paths and metadata first.
- Decode images lazily and inspect in bounded batches.
- Start with `num_workers=2`, `pin_memory=True` only with CUDA, and no persistent workers until verified.

**Exit check**

- A machine-readable audit report and a human-readable summary identify the authoritative label table and any data-quality risks.

## Stage 2 — Split and subset design

### Leakage prevention

Where filenames permit, derive a patient/group key and ensure all images from one patient remain in one split. Exact duplicate hashes or perceptual near-duplicates discovered during audit must also be grouped.

### Full-data split

Preferred design:

- preserve the original EyePACS train/test provenance when reliable;
- derive a validation split only from the original training portion;
- use group-aware stratification by DR grade;
- keep the original test portion untouched until final evaluation.

If provenance is ambiguous, create deterministic group-stratified train/validation/test manifests and document the deviation.

### Subset policy

Subset experiments are mandatory before full-scale training.

Candidate tiers:

- **Smoke:** approximately 50–100 images per class, capped by minority-class availability.
- **Development:** approximately 500–1,000 images per class, or a fixed total with class-aware sampling.
- **Extended:** largest subset that preserves minority classes and fits the runtime budget.
- **Full:** all eligible images.

Sampling rules:

- deterministic seed;
- group-aware selection;
- no oversampling before split creation;
- save CSV manifests with path, label, source split, group ID, and subset tier;
- save counts and a SHA-256 hash of every manifest.

**Exit check**

- No group or duplicate hash crosses splits, and every class has an acceptable count in each evaluation split.

## Stage 3 — Data pipeline

**Implementation sequence**

1. Dataset class returning image, ordinal label, identifier, and optional metadata.
2. Fundus preprocessing benchmark: resize/crop, optional circular crop, border removal, normalization.
3. Conservative augmentations appropriate for fundus images: horizontal flip, limited rotation, scale/crop, brightness/contrast, and mild color perturbation.
4. Validation/test transforms with no stochastic operations.
5. DataLoader diagnostics and a visual augmentation grid.

**Verification**

- Label-path alignment tests.
- Deterministic validation batches.
- Batch shape/range checks.
- Throughput and peak-memory measurement.
- No augmentation that destroys laterality or clinically relevant structure without explicit justification.

## Stage 4 — Baseline modelling

Start with a pretrained, efficient convolutional backbone, then compare stronger alternatives only after the baseline is stable.

**Baseline candidates**

- EfficientNet-B0/B2 or ConvNeXt-Tiny, depending on installed torchvision support and VRAM.
- Five-class softmax classification as the first objective.

**Later ordinal alternatives**

- CORAL/CORN-style ordinal heads;
- regression or cumulative-link formulations;
- hybrid classification plus ordinal-distance loss.

**Training safeguards**

- automatic mixed precision;
- gradient accumulation when batch size is constrained;
- gradient clipping;
- early stopping;
- best/last checkpointing;
- finite-loss checks;
- batch-level and epoch-level `tqdm` progress;
- controlled image-size schedule, beginning at 224 or 256 pixels;
- resume support.

**Class imbalance experiments**

Compare, one factor at a time:

- plain cross-entropy;
- inverse-frequency or effective-number class weights;
- focal loss;
- weighted sampling;
- ordinal-aware losses.

Weighted sampling and heavily weighted losses should not be combined initially because their effects can compound.

## Stage 5 — Evaluation

### Primary metric

- Quadratic weighted kappa (QWK), appropriate for ordered DR grades.

### Supporting metrics

- macro and weighted F1;
- balanced accuracy;
- one-vs-rest AUROC and AUPRC;
- per-class sensitivity, specificity, precision, recall, and support;
- exact accuracy and within-one-grade accuracy;
- multiclass confusion matrix, raw and normalized;
- calibration error, reliability diagram, Brier score, and optional temperature scaling;
- bootstrap confidence intervals with patient/group-level resampling where possible.

### Thresholding

Default predictions use the fixed argmax rule. Any threshold or ordinal-cutpoint optimization must be fit on validation data only and locked before test evaluation.

## Stage 6 — Explainability and qualitative analysis

**Grad-CAM suite**

- correct high-confidence examples from every class;
- incorrect high-confidence examples;
- adjacent-grade confusion cases;
- low-confidence/uncertain cases;
- paired original, heatmap, and overlay panels;
- consistent colormap, scale, labels, and resolution;
- configurable target layer and sanity checks.

**Additional analysis**

- occlusion sensitivity for selected cases;
- embedding visualization only as exploratory evidence;
- image-quality stratification;
- confidence and entropy distributions;
- error galleries by true/predicted grade;
- optional comparison with lesion-relevant regions when annotations exist.

Grad-CAM is a post-hoc visualization and must not be presented as validation of clinical reasoning.

## Stage 7 — Controlled experiments

Recommended order:

1. Smoke test on a tiny subset.
2. Baseline on development subset.
3. Image size comparison.
4. Preprocessing comparison.
5. Imbalance method comparison.
6. Backbone comparison.
7. Ordinal-objective comparison.
8. Test-time augmentation.
9. Ensemble only after individual models are stable.
10. Extended/full-data confirmation of the best small-scale recipe.

Each experiment changes one major factor and inherits a named base configuration.

## Stage 8 — Reproducibility package

Every completed run should export:

- resolved YAML configuration;
- environment snapshot;
- random seed;
- split and subset manifest hashes;
- Git commit SHA;
- training history CSV/JSON;
- predictions CSV;
- metrics JSON;
- figures;
- checkpoint metadata;
- concise model card describing data, intended use, limitations, and observed failure modes.

Large checkpoints should remain in Kaggle Outputs, a release asset, or external artifact storage rather than ordinary Git history.

## Stage 9 — Final repository

Final code organization will separate reusable modules from notebooks:

- `src/dr_detection/data`: manifests, datasets, transforms, samplers;
- `src/dr_detection/models`: backbones, heads, losses;
- `src/dr_detection/training`: loops, callbacks, checkpoints;
- `src/dr_detection/evaluation`: metrics, calibration, bootstrap, plots;
- `src/dr_detection/explainability`: Grad-CAM and qualitative panels;
- `src/dr_detection/utils`: seeds, logging, environment, paths;
- `configs`: composable experiment configurations;
- `scripts`: command-line entry points;
- `tests`: fast unit and integration tests;
- `notebooks`: thin, ordered Kaggle drivers.

## Step-by-step working rule

Only one stage is implemented at a time. A stage is merged after its notebook/script runs successfully, its outputs are inspected, and its tests pass. The next implementation task is **Stage 1: environment and dataset audit**.