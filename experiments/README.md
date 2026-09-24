# Link-prediction implementation

This directory contains the maintained training and inference workflows:

| Directory | Purpose |
| --- | --- |
| `gin/` | Main-table reproduction protocol, input verification, and deterministic evaluation support. |
| `semantic_mlp/` | Semantic GIN training, checkpoint scoring, and heuristic baselines. The historical module name is retained for checkpoint compatibility. |
| `llm_lp/` | LLM link-prediction evaluation, PEFT fine-tuning, and scoring saved binary prompts with Transformers or vLLM. |
| `tabicl/` | The router, train-context fusion, and table helper required by the online GNN/LLM evaluator. |
| `modules/` | Shared model, data, prompt, scoring, and runtime implementations. |
| `vllm_capture_bootstrap/` | Worker bootstrap for optional prompt-embedding capture by the main LLM evaluator. |

Run entrypoints from the repository root after installing their dependencies:

```bash
python -m experiments.semantic_mlp.train_semantic_mlp_pipeline --help
python -m experiments.llm_lp.evaluate_llm_link_prediction --help
python -m experiments.llm_lp.train_peft_link_prediction --help
```

See the [repository README](../README.md) for the main-table reproduction
commands, data setup, and supported environments. Optional prompt-vector
capture is described in [LLM_PROMPT_EMBEDDINGS.md](llm_lp/LLM_PROMPT_EMBEDDINGS.md).

The [LLM history protocol](../docs/llm-history-protocol.md) defines both-endpoint
incoming/outgoing histories, the default 47-event window, and directed
source-to-target interaction counts.

Local LLM evaluation and prompt export require an explicit `--model_path`;
pass your checkpoint directory or Hugging Face model identifier. PEFT training
also requires `--model_path` and defaults to `--train_data_protocol dtgb_strict`:
training queries, negative destinations, and prompt history use the observed
training graph. `--train_data_protocol legacy_time_only` is available only for
reproducing older training runs. Use a fresh `--output_dir` for a new strict run;
strict checkpoint resumption verifies the saved graph and sample provenance.

Training workflows default to five independent runs. Evaluation and TabICL
ensembles also default to five trials or ensemble members.

The semantic and PEFT training entrypoints launch a separate process per seed
by default. PEFT outputs go to `<output_dir>/seed-<seed>/`; semantic checkpoints
go to `<checkpoint_parent>/seed-<seed>/<checkpoint_name>`. See each entrypoint's
`--help` for single-run overrides and checkpoint resumption.

For example, train five independent Llama adapters with:

```bash
python -m experiments.llm_lp.train_peft_link_prediction --dataset_name Enron --model_path /path/to/base-model --output_dir outputs/llama/enron
```

Evaluate each adapter's model separately with its matching `--seed` and
`--num_trials 1`. The evaluation entrypoint defaults to five evaluation trials;
these trials reuse one supplied model and
do not count as independent fine-tuning runs. Report mean and sample standard
deviation (`ddof=1`) across independent training runs, keeping both full test
splits per run.

Keep generated metrics, prediction files, adapters, and checkpoints outside
Git. The repository provides code and configurations for generating these
artifacts locally.

Install `requirements-llm.txt` for Transformers/PEFT workflows. Optional backends
need their own dependencies: vLLM for accelerated inference, TabICL for online
routing/fusion, `openai` for API evaluation, and `bitsandbytes` for 4-bit PEFT.
Keep vLLM and TabICL in separate environments as required by their PyTorch
versions. The main GIN recipe does not use these optional backends.
