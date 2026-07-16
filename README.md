# Diabetic Retinopathy Detection — EyePACS / PyTorch

A reproducible, memory-safe PyTorch research pipeline for grading diabetic retinopathy from retinal fundus images using the pre-resized EyePACS dataset on Kaggle.

> **Research use only.** This repository is not a medical device and must not be used for clinical diagnosis or treatment decisions.

## Current status

**Phase 0: repository design and experiment planning**

- [x] Define repository structure and development workflow
- [x] Define dataset audit, subset, split, training, evaluation, and explainability plan
- [ ] Implement and verify the Kaggle data-audit notebook
- [ ] Build a deterministic stratified subset
- [ ] Train baseline models
- [ ] Run controlled experiments and ablations
- [ ] Produce Grad-CAM, error-analysis, calibration, and publication-quality figures

No training implementation is committed yet. Development will proceed step by step in Kaggle, with each stage tested before the next is added.

## Dataset paths on Kaggle

```text
/kaggle/input/datasets/mohlamin/resized-eyepacs-diabetic-retinopathy-dataset/Images
/kaggle/input/datasets/mohlamin/resized-eyepacs-diabetic-retinopathy-dataset/all_labels.csv
/kaggle/input/datasets/mohlamin/resized-eyepacs-diabetic-retinopathy-dataset/original_test_labels.csv
/kaggle/input/datasets/mohlamin/resized-eyepacs-diabetic-retinopathy-dataset/original_train_labels.csv
```

The dataset is never committed to GitHub. Generated checkpoints, caches, predictions, and large figures are also excluded.

## Planned repository layout

```text
DR_Detection/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── data/
│   ├── model/
│   ├── train/
│   └── experiment/
├── docs/
│   ├── PROJECT_PLAN.md
│   ├── EXPERIMENT_PROTOCOL.md
│   └── KAGGLE_GITHUB_WORKFLOW.md
├── notebooks/
│   ├── 00_environment_and_data_audit.ipynb
│   ├── 01_subset_and_split_verification.ipynb
│   ├── 02_baseline_training.ipynb
│   ├── 03_evaluation_and_error_analysis.ipynb
│   └── 04_gradcam_and_explainability.ipynb
├── src/dr_detection/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── explainability/
│   └── utils/
├── scripts/
├── tests/
├── reports/figures/
└── artifacts/
```

Folders are introduced as their corresponding implementation stage begins, preventing untested boilerplate from accumulating.

## Development principles

1. **Patient-aware and leakage-resistant validation.** Left/right eye pairs and duplicate images must not cross data splits when identity can be inferred.
2. **Deterministic experiments.** Seeds, package versions, configurations, split manifests, and commit hashes are recorded.
3. **Memory safety first.** Dataset metadata is audited before image decoding; loaders use conservative workers, mixed precision, gradient accumulation, and staged image sizes.
4. **Metric alignment.** Primary model selection uses quadratic weighted kappa, supported by macro F1, balanced accuracy, per-class sensitivity/specificity, ROC/PR analysis, confusion matrices, and calibration.
5. **Clinical caution.** Explainability maps are treated as model-behaviour visualizations, not proof of clinical reasoning.
6. **Reproducibility.** Every result must be recoverable from a committed configuration and a documented Kaggle run.

## Authentication from Kaggle

Use the existing Kaggle secret named `pushDR`; never print or commit its value.

```python
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
github_token = user_secrets.get_secret("pushDR")
```

The token is used only in the notebook runtime to authenticate a Git push. See `docs/KAGGLE_GITHUB_WORKFLOW.md` for the safe workflow.

## Planned progress reporting

Long-running stages will expose progress bars through `tqdm`, including:

- image-file indexing and integrity checks;
- metadata/image matching;
- subset construction;
- training and validation batches;
- test-time inference and TTA;
- Grad-CAM generation;
- bootstrap confidence intervals and error-analysis exports.

Progress bars will be disabled automatically in non-interactive test contexts.

## Reproducibility target

Each experiment will save:

- the resolved configuration;
- random seed and split-manifest hash;
- Git commit SHA;
- environment/package snapshot;
- epoch-level metrics;
- best and last checkpoints;
- predictions with image identifiers;
- evaluation plots and explainability outputs.

## Author

GitHub: `itsCodeBakery`

Contact: `ashayan29@gmail.com`
