# Reproduce the GIN main table

The public entry point is `scripts/reproduce_gin.py`. It runs the audited rolling-snapshot GIN with E5-large-v2 entity embeddings and four heuristics (`recency,popularity,past,ra`), with MPLP disabled. It needs no W&B account or machine-specific directory layout.

The default reproduction protocol uses five distinct training seeds: `42, 43, 44, 45, 46`. Run the commands below to generate per-seed results and their mean and sample standard deviation. Dataset configurations and input fingerprints are included in the repository; generated results, logs, and checkpoints remain local.

## Install and prepare data

From the repository root, create a Python environment and install [the core requirements](../requirements.txt). The reference environments used Python 3.10 and CUDA GPUs. Install the PyTorch build appropriate to your CUDA driver before the remaining dependencies.

Download the DTGB datasets using the link in the [repository README](../README.md). Set up an external data directory with these files; the data and caches do not belong in Git:

```text
/path/DyLink_Datasets/
  GDELT/{edge_list.csv,entity_text.csv}
  ICEWS1819/{edge_list.csv,entity_text.csv}
  Enron/{edge_list.csv,entity_text.csv}
  Googlemap_CT/{edge_list.csv,entity_text.csv}
```

`edge_list.csv` uses columns `u,r,i,ts,label`; `entity_text.csv` uses `i,text`. The frozen loader implements the chronological 70%/15%/15% split, the historical inductive-node holdout with data seed 2020, and the GDELT timestamp conversion. Keep the original file contents and row order. The training seed does not redefine the data split.

Provide an existing dataset-specific E5-large-v2 NumPy cache, or generate one:

```bash
python scripts/prepare_gin_embeddings.py --dataset GDELT --data-root /path/DyLink_Datasets --output /path/cache/GDELT.npy --device cuda:0
python scripts/prepare_gin_embeddings.py --dataset ICEWS1819 --data-root /path/DyLink_Datasets --output /path/cache/ICEWS1819.npy --device cuda:0
python scripts/prepare_gin_embeddings.py --dataset Enron --data-root /path/DyLink_Datasets --output /path/cache/Enron.npy --device cuda:0
python scripts/prepare_gin_embeddings.py --dataset Googlemap_CT --data-root /path/DyLink_Datasets --output /path/cache/Googlemap_CT.npy --device cuda:0
```

The helper reuses the frozen trainer's preprocessing: raw entity text for GDELT, Enron, and Googlemap_CT; compressed entity names for ICEWS1819; E5 `passage:` prefix, normalized 1024-dimensional embeddings, and rows sorted by entity ID. The 1024 dimensions are propagated without a learned semantic projection. `--model /path/e5-large-v2` uses a local model directory. Each cache has a metadata sidecar with its entity ordering and input/output hashes.

The original caches are identified by SHA-256 in `configs/gin/*.json`; they are not bundled in this code repository. The original model-download revision was not recorded, so regeneration is not a promise of byte-identical cache contents. Hardware and encoder-library changes can also change floating-point results. The training command requires reference input hashes by default. If a regenerated cache differs, review the metadata and deliberately use `--allow-input-mismatch`; the manifest and aggregate then identify it as a new input variant. Do not present such a run as an exact replay of the archived caches.

## Check and run the recipes

Preview the effective trainer arguments without importing PyTorch, downloading anything, creating outputs, or training:

```bash
python scripts/reproduce_gin.py run --dataset GDELT --data-root /path/DyLink_Datasets --embedding-cache /path/cache/GDELT.npy --seeds 42 43 44 45 46 --dry-run
```

Run each main-table dataset, substituting your own input paths:

```bash
python scripts/reproduce_gin.py run --dataset GDELT --data-root /path/DyLink_Datasets --embedding-cache /path/cache/GDELT.npy --seeds 42 43 44 45 46 --gpu 0
python scripts/reproduce_gin.py run --dataset ICEWS1819 --data-root /path/DyLink_Datasets --embedding-cache /path/cache/ICEWS1819.npy --seeds 42 43 44 45 46 --gpu 0
python scripts/reproduce_gin.py run --dataset Enron --data-root /path/DyLink_Datasets --embedding-cache /path/cache/Enron.npy --seeds 42 43 44 45 46 --gpu 0
python scripts/reproduce_gin.py run --dataset Googlemap_CT --data-root /path/DyLink_Datasets --embedding-cache /path/cache/Googlemap_CT.npy --seeds 42 43 44 45 46 --gpu 0
```

`--gpu` is the logical device after `CUDA_VISIBLE_DEVICES`. Each seed runs in its own process. To run seeds on separate devices, invoke the command separately with `--seeds 42` through `--seeds 46`. To run a subset, pass its seed list, such as `--seeds 45 46`. `--gpu -1` selects CPU, but large reference datasets are intended for GPU execution.

Output defaults to `outputs/gin/<dataset>/seed-<seed>/`. Existing seed directories cause an error; use a new `--output-root` for another experiment. Failed runs keep their diagnostic files and do not count toward aggregation. The runner never overwrites or deletes an old checkpoint.

| Dataset | History | GIN layers | Scorer hidden / layers | LR | Dropout | Fusion | Evaluation batch | Budget |
| --- | --- | ---: | --- | ---: | ---: | --- | ---: | --- |
| GDELT | W = 10 | 1 | 128 / 4 | 5e-5 | 0 | late concatenation | 128 | max 50, patience 5 |
| ICEWS1819 | W = 10 | 1 | 256 / 2 | 1e-4 | 0 | late concatenation | 256 | max 50, patience 5 |
| Enron | K = 200 | 2 | 64 / 2 | 1e-4 | 0.2 | residual | 256 | full 50, no early stopping |
| Googlemap_CT | K = 200 | 3 | 256 / 2 | 5e-5 | 0 | late concatenation | 256 | max 50, patience 5 |

All remaining settings are explicit in [the four JSON recipes](../configs/gin). W mode sets effective K to zero; K mode sets the effective time window to `1e15`. An inactive sweep parameter does not impose an additional history cutoff. Preserve the evaluation batch size: it affects rolling graph refresh and heuristic normalization, not just throughput. The current Enron recipe is full 50 epochs; earlier 10-epoch experiments are a separate result lineage.

## Evaluation and aggregation

The runner selects the checkpoint by validation **AUC**, restores every learned component, and evaluates each final test split exactly once. It does not select on test results or evaluate the tests after every epoch. One random negative accompanies each positive query. Evaluation negatives are constructed using canonical groups of 256 and replayed in fixed query order, independently of the model's evaluation batch size. Precomputed heuristic negatives and evaluated negatives are checked for agreement.

The reported main metric is the DTGB mean of AUC values in canonical groups of 256 positive queries (`test/transductive_auc` and `test/inductive_auc`). Pooled/global AUC is logged separately and is not substituted into the table. Reference input files also require matching query and negative-table fingerprints.

After all five seeds finish:

```bash
python scripts/reproduce_gin.py summarize --output-root outputs/gin
```

The command reports raw per-seed AUC, the arithmetic mean, and **sample** standard deviation (`ddof=1`). It rejects missing seeds, duplicate seeds, smoke runs, changed inputs, and incompatible source/configuration/evaluation populations. To inspect an unfinished experiment without implying completion:

```bash
python scripts/reproduce_gin.py summarize --output-root outputs/gin --datasets GDELT --allow-partial
```

`--allow-partial` reports the actual sample count and `five_seed_complete: false`, plus the target and missing seed lists. It does not invent missing values. A single completed seed has `sample_std: null`. Add `--allow-input-mismatch` only when intentionally summarizing an input variant; incompatible variants still cannot be mixed across seeds.

Every successful seed directory contains the chosen `best.pt`, `effective_config.json`, a manifest with source/input hashes and environment versions, `metrics.jsonl`, canonical negative tables, `timing.json`, and `result.json`. The source-preserving adapter in `experiments/gin/protocol.py` applies the audited evaluation rules to the frozen model code. `configs/gin/source-pins.json` checks the retained implementation and the shared seed dispatcher. Reviewed changes to entrypoint seed defaults and TabICL ensemble size are pinned alongside the model code. Two historical unused TabICL files were omitted, and the unrelated baseline configuration loader retains its baseline version. Source and input verification runs locally before training.

For an installation smoke check on a small local dataset, use `--seeds 42 --allow-input-mismatch --smoke-test` and a fresh output directory. This performs one epoch and final checkpoint restoration, marks the result `smoke_test`, and prevents it from entering the main-table aggregate. It does not reproduce the reference metrics.
