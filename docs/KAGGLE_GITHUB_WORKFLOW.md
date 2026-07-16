# Kaggle-to-GitHub Workflow

## Secret handling

The GitHub token is stored in Kaggle Secrets under the name `pushDR`.

```python
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
github_token = user_secrets.get_secret("pushDR")
assert github_token, "Missing Kaggle secret: pushDR"
```

Never print the token, place it in a notebook cell output, save it in a configuration file, embed it in a Git remote that will be displayed, or commit it.

## Runtime Git identity

```bash
git config --global user.name "itsCodeBakery"
git config --global user.email "ashayan29@gmail.com"
```

## Recommended notebook workflow

1. Clone the repository into `/kaggle/working`.
2. Checkout or create a small feature branch for one verified stage.
3. Run the notebook and tests.
4. Clear large notebook outputs before committing.
5. Review `git status` and `git diff`.
6. Commit source, configuration, documentation, and small result summaries only.
7. Push the branch using the runtime token.
8. Merge only after the stage is reproducible.

## Safe clone and push pattern

Use Python to provide credentials without printing them:

```python
import os
import subprocess
from kaggle_secrets import UserSecretsClient

repo = "https://github.com/itsCodeBakery/DR_Detection.git"
token = UserSecretsClient().get_secret("pushDR")
env = os.environ.copy()
env["GITHUB_TOKEN"] = token

# GitHub accepts the token through an HTTP authorization header.
header = f"AUTHORIZATION: bearer {token}"
subprocess.run(
    ["git", "-c", f"http.extraHeader={header}", "clone", repo],
    cwd="/kaggle/working",
    env=env,
    check=True,
)
```

For a push from the cloned directory:

```python
subprocess.run(
    ["git", "-c", f"http.extraHeader={header}", "push", "origin", "HEAD"],
    cwd="/kaggle/working/DR_Detection",
    env=env,
    check=True,
)
```

Do not expose `header`, `env`, or the complete command in notebook output.

## What belongs in GitHub

Commit:

- source code and tests;
- compact notebooks with cleared outputs;
- YAML/TOML configurations;
- split-generation logic;
- small split manifests when licensing and size permit;
- metrics JSON/CSV summaries;
- selected optimized figures;
- environment and reproduction instructions.

Do not commit:

- EyePACS images or supplied CSV files unless redistribution is explicitly permitted;
- model checkpoints and optimizer states;
- cache directories;
- raw predictions containing unnecessary sensitive metadata;
- full experiment directories;
- credentials, tokens, `.env` files, or authenticated URLs.

## Large artifacts

Store large outputs in Kaggle notebook outputs or another artifact store. Record their checksum, producing commit SHA, configuration, and retrieval location in a small metadata file. Git LFS may be considered later, but ordinary Git history must remain lightweight.

## Notebook hygiene

Before committing:

- restart and run all relevant cells;
- remove token-bearing cells from output history;
- clear bulky image grids and progress-bar output;
- ensure hard-coded paths are centralized in configuration;
- retain only concise evidence needed to understand the run;
- export reusable logic into `src/dr_detection` rather than duplicating it across notebooks.

## Commit convention

Use small, descriptive commits:

```text
docs: define dataset audit protocol
feat(data): add manifest validation
feat(train): add mixed-precision training loop
test(data): cover grouped split leakage
fix(eval): compute QWK with fixed class range
```
