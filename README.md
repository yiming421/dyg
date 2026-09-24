# GIN and LLM Link Prediction on DTGB

Research code for temporal link prediction with a lightweight semantic GIN,
LLM predictors, and reusable fusion components. This repository is based on
[DTGB](https://github.com/zjs123/DTGB) and [DyGLib](https://github.com/yule-BUAA/DyGLib).

**Start with [GIN reproduction](docs/GIN_REPRODUCTION.md).** It contains the
four dataset recipes, input fingerprints, seed policy, validation-selected
checkpoint protocol, and commands to regenerate the GIN row of the main table.
The default reproduction protocol uses **five seeds**. GIN, Llama training
and evaluation, and TabICL alignment use `42, 43, 44, 45, 46`; temporal graph
baselines retain `0, 1, 2, 3, 4`, and TabICL routers use `142, 143, 144, 145, 146`.
Report the mean and sample standard deviation (`ddof=1`) across independent
training runs. TabICL ensembles average predictions across their five seeds.

This repository contains implementation code, experiment configurations, and
reproduction instructions. Generated metrics, per-run records, logs, and model
checkpoints are stored outside Git. Repeated evaluations of one checkpoint
should be distinguished from independent training runs.

## Setup

The GIN reference environment uses Linux, Python 3.10 and PyTorch 2.3.0.
Create a separate environment and install the requirements:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a PyTorch build compatible with your CUDA runtime when using a GPU.
The optional LLM training dependencies are in `requirements-llm.txt`.
GIN reproduction does not require a W&B account.

## Data

Download the datasets from the
[original DTGB data release](https://drive.google.com/drive/folders/1QFxHIjusLOFma30gF59_hcB19Ix3QZtk).
The GIN recipes use `GDELT`, `ICEWS1819`, `Enron`, and `Googlemap_CT`.
Each directory must contain `edge_list.csv` and `entity_text.csv`:

```text
/path/to/DyLink_Datasets/
  GDELT/
  ICEWS1819/
  Enron/
  Googlemap_CT/
```

Edges use the DTGB fields `u,r,i,ts,label`. Entity text maps entity IDs to
descriptions. The GIN recipes use E5-large-v2 entity embeddings; follow the
preparation and hash checks in the reproduction guide. Datasets, embeddings
and model weights are kept outside Git.

## Reproduce the main GIN experiment

Preview a recipe without loading Torch, data, or allocating a GPU:

```bash
python scripts/reproduce_gin.py run \
  --dataset Enron \
  --data-root /path/to/DyLink_Datasets \
  --embedding-cache /path/to/embeddings/enron_e5.npy \
  --output-root outputs/gin \
  --seeds 42 43 44 45 46 --dry-run
```

Remove `--dry-run` to train and evaluate. Replace the dataset and embedding
cache to run another recipe. The checked-in configs fix the history policy,
GIN architecture, heuristic features, evaluation batching and training budget.
See [the full guide](docs/GIN_REPRODUCTION.md) for evaluation and aggregation; the plain legacy semantic trainer is not the main-table entrypoint.

## Code map

| Path | Purpose |
| --- | --- |
| `scripts/reproduce_gin.py` | Main GIN training, evaluation and aggregation CLI |
| `configs/gin/` | Dataset recipes and source/input fingerprints |
| `experiments/gin/` | Reproduction protocol and deterministic evaluation support |
| `experiments/semantic_mlp/` | Semantic GIN trainer and scoring utilities |
| `experiments/llm_lp/` | LLM link prediction evaluation and PEFT training |
| `experiments/tabicl/` | Reusable table, router and fusion entrypoints |
| `experiments/modules/` | Shared model, prompt, graph and evaluation modules |
| `models/`, `utils/` | Temporal graph baselines and DTGB data utilities |
| `docs/` | Reproduction instructions |

[Experiments documentation](experiments/README.md) lists the supported
entrypoints. The retained `train_link_prediction.py` is the original DTGB
baseline interface; its CLI defaults do not reproduce the new GIN table.
Historical baseline variants need their own recorded configuration and source.

## Acknowledgements

This repository builds upon the [DTGB](https://github.com/zjs123/DTGB) codebase.
We extend it with our experiments. We thank the DTGB authors for releasing
their code and datasets.

If you use this repository, please also cite the original DTGB paper.

DTGB: A Comprehensive Benchmark for Dynamic Text-Attributed Graphs,
Zhang et al., NeurIPS 2024.
[Paper](https://arxiv.org/abs/2406.12072).

```bibtex
@article{zhang2024dtgb,
  title={DTGB: A Comprehensive Benchmark for Dynamic Text-Attributed Graphs},
  author={Zhang, Jiasheng and Chen, Jialin and Yang, Menglin and Feng, Aosong and Liang, Shuang and Shao, Jie and Ying, Rex},
  journal={arXiv preprint arXiv:2406.12072},
  year={2024}
}
```
