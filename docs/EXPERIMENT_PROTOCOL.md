# Experiment Protocol

## Fixed principles

- Never select a model using test-set performance.
- Never allow patient groups, left/right eye pairs, or detected duplicates to cross splits.
- Never alter multiple major variables in one ablation.
- Never report only accuracy for this imbalanced ordinal task.
- Never commit secrets, raw data, large checkpoints, or notebook outputs containing tokens.

## Reproducibility controls

Every run receives a unique run ID and records:

- UTC timestamp and Kaggle notebook/kernel identifier;
- Git commit SHA and dirty-state flag;
- Python, PyTorch, torchvision, CUDA, cuDNN, and GPU details;
- all seeds and deterministic settings;
- complete resolved configuration;
- data manifest path and SHA-256 hash;
- trainable parameter count;
- peak CPU/GPU memory and runtime;
- best-epoch selection rule.

Determinism will be enabled where practical. Any nondeterministic CUDA operation retained for performance must be documented.

## Split contract

The split-generation step must emit immutable CSV manifests. Required columns are:

```text
image_id, image_path, label, source_split, group_id, split, subset_tier
```

Optional audit columns include file size, width, height, hash, image-quality flags, and laterality.

Before training, assertions must prove:

- each image identifier appears once;
- labels belong to `{0,1,2,3,4}`;
- all paths exist;
- group IDs are disjoint across splits;
- duplicate hashes are disjoint across splits;
- train/validation/test counts and class distributions match the saved report.

## Training contract

Each epoch logs at least:

- train loss;
- validation loss;
- learning rate;
- QWK;
- macro F1;
- balanced accuracy;
- exact accuracy;
- within-one-grade accuracy;
- epoch duration;
- peak allocated/reserved GPU memory.

Batch progress bars display running loss and learning rate without retaining batch tensors. Validation/inference progress bars avoid expensive per-batch metric recomputation; predictions are accumulated on CPU and scored once per epoch.

## Model selection

The default best checkpoint is the epoch with highest validation QWK. Ties are resolved by higher macro F1, then lower validation loss. Early stopping uses validation QWK with a declared patience and minimum delta.

## Evaluation contract

Test evaluation occurs once for a finalized experiment family. It exports:

- per-image logits, probabilities, predictions, labels, confidence, entropy, and identifiers;
- a metrics JSON file;
- raw and normalized confusion matrices;
- class-wise metric table;
- ROC and precision-recall figures where statistically meaningful;
- calibration/reliability figure;
- bootstrap confidence intervals;
- an error-analysis table and galleries.

Confidence intervals should resample patient/group IDs rather than individual eyes when group IDs are available.

## Figure-quality standard

Figures should be publication-ready:

- vector output (`.svg` or `.pdf`) where suitable, plus high-resolution PNG;
- readable typography and labels;
- descriptive titles and captions;
- no clipped axes or legends;
- consistent class names and grade ordering;
- colorblind-conscious palettes when colors are introduced;
- exact sample counts shown for class-wise plots;
- deterministic selection criteria for qualitative examples.

## Grad-CAM protocol

Grad-CAM outputs must record model checkpoint, target class, target layer, preprocessing, and image identifier. Panels should include the original image, normalized heatmap, overlay, true grade, predicted grade, and confidence.

Quality checks:

- verify hooks are removed after use;
- run inference in evaluation mode;
- preserve gradients only for the target forward/backward pass;
- reject constant or non-finite maps;
- compare maps for predicted and true classes on selected errors;
- include sanity checks such as randomized final-layer weights on a small sample when feasible.

## Ablation table

Maintain one row per experiment with:

```text
run_id, commit_sha, subset, backbone, image_size, preprocessing, augmentation,
loss, sampler, optimizer, scheduler, epochs, best_epoch, val_qwk, val_macro_f1,
test_qwk, test_macro_f1, runtime, peak_vram, notes
```

Test columns remain empty until a model family is locked.

## Failure handling

- On CUDA OOM: clear references, record the failed configuration, reduce batch size, then use gradient accumulation before reducing image size.
- On data corruption: quarantine the file in a manifest rather than silently skipping it.
- On NaN/Inf: stop immediately, save diagnostic metadata, and inspect inputs, loss scaling, and learning rate.
- On interrupted training: resume only from a checkpoint with matching configuration and manifest hash.
