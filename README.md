# 🤯 OverThink: Slowdown Attacks on Reasoning LLMs

## 💡 Introduction 
This is an official repository of our paper "*OverThink: Slowdown Attacks on Reasoning LLMs*". In this attack, we aim to increase the number of reasoning tokens without manipulating the generated output. 

Please follow the steps below to test our **OverThink** attack.

[Paper Print](https://arxiv.org/pdf/2502.02542)

[Dataset](https://huggingface.co/datasets/akumar0927/OverThink)

# OverThink updated files

This folder holds the scripts used to (1) compile attack datasets, (2) run the context-agnostic
attack on the Hugging Face dataset, and (3) evolve adversarial templates that maximize
reasoning-token usage.

## What is in this folder

- `compile_datasets.py`: Build FreshQA, SQuAD, and MuSR attack CSVs from source data.
- `context_agnostic_hf.py`: Run context-agnostic attacks on HF dataset splits.
- `icl_evolve.py`: Evolve verbalized attack templates with a genetic search loop.
- `utils.py`: Provider wrappers (OpenAI, Anthropic, Mistral, Fireworks, Google) and .env loading.
- `dataset/`: Generated CSVs (freshQA_attack.csv, squad_attack.csv, MuSR/*).
- `FreshQA_v12182024 - freshqa.csv`: FreshQA source CSV (used by some scripts).

## Setup

### Python packages
These scripts assume a Python environment with:

- Core: `pandas`, `datasets`, `requests`, `beautifulsoup4`, `tqdm`
- Models: `openai`, `anthropic`, `mistralai`, `google-genai`
- Plotting: `matplotlib`
- Optional token counting: `tiktoken`

Only install the model packages you plan to use.

### Environment variables

`utils.py` loads `.env` from the repo root (`OVERTHINK/.env`). `icl_evolve.py` loads `.env`
next to the script (`public_repo_update/.env`). Make sure the needed keys are present in the
right file for the script you run.

Common keys:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `MISTRAL_API_KEY`
- `FIREWORKS_API_KEY`
- `GEMINI_API_KEY`

## Datasets

### FreshQA source CSV

Scripts expect a FreshQA CSV with at least these columns:
- `question`
- `source` (Wikipedia URL(s), newline-separated)
- `answer_0`
- `fact_type` (used by `icl_evolve.py`; values like `none-changing` or `slow-changing`)

The included file is `public_repo_update/FreshQA_v12182024 - freshqa.csv`. The default
for `compile_datasets.py` points to `s&p_submission_exp/FreshQA_v12182024 - freshqa.csv`,
so override with `--freshqa-csv` if you want to use the local copy.

### Attack CSV format

`compile_datasets.py` writes CSVs with these columns:
- `Source`: base prompt
- `Answer`: ground-truth answer (FreshQA and MuSR)
- `Attack_Source_1` ... `Attack_Source_7`: attack prompts

These CSVs are used to build the HF dataset splits (e.g., `freshQA_attack`).

## Scripts

### 1) `compile_datasets.py`

Builds the FreshQA, SQuAD, and MuSR attack CSVs using 7 built-in templates.

Example:

```bash
python public_repo_update/compile_datasets.py \
  --freshqa-csv "public_repo_update/FreshQA_v12182024 - freshqa.csv" \
  --output-dir public_repo_update/dataset \
  --freshqa-limit 100 \
  --squad-limit 100 \
  --musr-limit 50
```

Arguments:
- `--freshqa-csv`: path to FreshQA CSV.
- `--output-dir`: output directory for CSVs.
- `--freshqa-limit`: number of FreshQA rows.
- `--squad-limit`: number of SQuAD rows (from validation split).
- `--musr-limit`: number of MuSR rows per split.
- `--no-fetch`: skip Wikipedia fetching; use raw `source` strings instead.
- `--skip-freshqa`, `--skip-squad`, `--skip-musr`: skip building specific datasets.

Outputs:
- `dataset/freshQA_attack.csv`
- `dataset/squad_attack.csv`
- `dataset/MuSR/murder_mystery.csv`
- `dataset/MuSR/object_placement.csv`
- `dataset/MuSR/team_allocation.csv`

### 2) `context_agnostic_hf.py`

Runs the context-agnostic attack on the Hugging Face dataset splits and saves responses
as a pickle after each row. It also prints running averages of reasoning tokens per source.

Example:

```bash
python public_repo_update/context_agnostic_hf.py \
  --split freshQA_attack \
  --provider OpenAI \
  --model o1-preview \
  --output-file freshqa_hf.pkl
```

Arguments:
- `--dataset-name`: HF dataset name (default `akumar0927/OverThink`).
- `--split`: HF split name (e.g., `freshQA_attack`, `squad_attack`,
  `MuSR_murder_mystery`, `MuSR_object_placement`, `MuSR_team_allocation`).
- `--model`: model passed to the provider.
- `--provider`: `OpenAI`, `Anthropic`, `Mistral`, `Firework`, or `Google`.
- `--output-file`: pickle path for results.
- `--start-index`: row index to start from.
- `--limit`: max number of rows to process.
- `--num-attacks`: how many `Attack_Source_*` columns to use.

MuSR aliases accepted for `--split`:
- `murder_mystery_dataset`, `object_placement_dataset`, `team_allocation_dataset`

Notes:
- Saves the pickle after each row so runs can resume.
- Token counting uses provider metadata when available. For providers that do not
  return reasoning tokens, the script falls back to `tiktoken` token counting if installed.

### 3) `icl_evolve.py`

Runs a verbalized genetic search to generate high-complexity prompt injections and scores
them by reasoning tokens. Produces a CSV of challenges and a score trajectory plot.

Example:

```bash
python public_repo_update/icl_evolve.py \
  --freshqa-csv "public_repo_update/FreshQA_v12182024 - freshqa.csv" \
  --top-p 0.6 \
  --k 5 \
  --epochs 15 \
  --score-model o3-mini \
  --generator-model o3-mini
```

Arguments:
- `--top-p`: nucleus sampling threshold for selecting candidate prompts.
- `--k`: number of prompts sampled per epoch from the nucleus distribution.
- `--epochs`: number of evolution rounds.
- `--score-model`: model used to score reasoning tokens.
- `--generator-model`: model used to generate new templates.
- `--repeats`: scoring repeats per template (averaged).
- `--sample-index`: which FreshQA row to use during scoring (from filtered subset).
- `--freshqa-csv`: FreshQA CSV path.
- `--output-dir`: directory for CSV and plot outputs.
- `--no-fetch`: skip Wikipedia fetching; use raw `source` instead.
- `--seed`: RNG seed for reproducibility.

Outputs (under `--output-dir`):
- `icl_evolve_samples_top_p_<top_p>_k_<k>.csv`
- `icl_evolve_score_trajectory_top_p_<top_p>_k_<k>.png`

## utils.py

`utils.py` provides helper functions for each model provider and handles loading
API keys from `.env`. It is imported by `context_agnostic_hf.py`.

If you add new providers, keep the `provider` names in `context_agnostic_hf.py`
in sync with the wrapper functions in `utils.py`.
