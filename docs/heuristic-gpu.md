# GPU heuristic correctness and reuse

The shared CN/AA/RA CUDA kernel counts each common neighbor once, matching the
CPU implementation. Repeated historical events still contribute to degree
weights. An equal-ID group qualifies if any event strictly precedes the query;
timestamps within an ID group need not be sorted. This reuses the fix verified
on September 19, 2026.

Run GPU experiments through the checked entry point:

```bash
python scripts/with_numba_cuda.py --cuda-home /path/to/cuda-toolkit TRAINING_SCRIPT [arguments]
```

It preserves CUDA_VISIBLE_DEVICES, configures isolated compiler libraries and
executes the real CPU/GPU/oracle regression before training. Missing CUDA or
failed equivalence stops the launch. The regression covers CN, AA and RA,
duplicate groups spanning multiple threads, unordered timestamps, strict time
boundaries, self queries, invalid nodes and empty inputs.

For train_link_prediction.py use --dygformer_use_heuristics
--dygformer_heuristic_scope full --dygformer_gpu_heuristics. Only common-neighbor
scoring is accelerated; sequence-history extraction retains its CPU path.

Old runs that actually used the duplicate-counting GPU kernel require retraining.
Runs using CPU common-neighbor scoring are unaffected by this kernel correction.
Do not resume an old GPU checkpoint or mix its results with corrected runs.
Keep historical model recipes, but apply this correctness patch and regression
to every source snapshot that will use the CUDA kernel.
