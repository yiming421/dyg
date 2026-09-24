# Contextual LLM prompt embeddings

`evaluate_llm_link_prediction.py` can optionally save one contextual vector for
each link-prediction prompt evaluated by the local vLLM backend:

```bash
conda run -n vllm python -m experiments.llm_lp.evaluate_llm_link_prediction \
  --dataset_name GDELT \
  --model_path /path/to/model \
  --capture_prompt_embeddings \
  --output results/llm_eval.json
```

By default, each vector is the raw final-RMSNorm state for the final prompt
token: the exact vector consumed by the LM head to predict the first answer
token. For forced-binary scoring, it represents the actual scoring prompt,
including the appended answer prefix. NPZ shards are written to
`results/llm_eval_prompt_embeddings/` unless
`--prompt_embedding_output_dir` is set. Each shard contains:

- `embeddings`: `[num_prompts, hidden_size]`, stored as float16 by default;
- link alignment fields: `query_id`, `source_id`, `target_id`, `relation_id`,
  `timestamp`, `dtgb_timestamp`, and `label`;
- `prompt_sha256`, `layer_id`, `normalized`, and representation metadata.

The result JSON records every shard path under `prompt_embedding_shards`.
Use `--prompt_embedding_save_dtype float32` for full-precision storage or
`--prompt_embedding_normalize` for optional L2 normalization. Explicit
`--prompt_embedding_layer N` values select vLLM's pre-decoder-block auxiliary
states; the recommended default `-1` selects the post-final-RMSNorm LM-head
input.

## Performance behavior

This mode keeps ordinary token generation batched in vLLM and does not run a
second Transformers forward pass. The custom connector transfers and writes
only one hidden-size vector per request, once. It does, however, activate
vLLM's `extract_hidden_states` execution path and disables chunked prefill, so
some throughput loss is expected. Runs without `--capture_prompt_embeddings`
use the original vLLM configuration and have no capture overhead.

When the parent process has already touched CUDA, launch vLLM with
`VLLM_WORKER_MULTIPROC_METHOD=spawn`; the default `fork` mode can inherit an
invalid CUDA runtime state in the engine child.

The saved matrix is intended as an optional input to a learned adapter and
residual correction over a GNN link representation. Keeping it out of the result JSON
also makes it straightforward to cache prompt vectors and train the fusion
module without rerunning the LLM.
