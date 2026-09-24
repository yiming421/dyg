# TabICL support for GNN/LLM inference

The maintained LLM evaluator uses
`experiments/modules/tabicl/online_pipeline.py` to launch routing and fusion in
a separate TabICL environment. This directory retains its required entrypoints:

| Entrypoint | Purpose |
| --- | --- |
| `routing/train_tabicl_utility_router.py` | Fit the router from labeled train context and score deployment rows. |
| `evaluation/evaluate_tabicl_train_llm_fusion.py` | Fit train-context fusion and evaluate selected splits. |
| `tables/build_tabicl_train_llm_table.py` | Join routed LLM diagnostics; also supplies the router's shared table helper. |

```bash
python -m experiments.tabicl.routing.train_tabicl_utility_router --help
python -m experiments.tabicl.evaluation.evaluate_tabicl_train_llm_fusion --help
python -m experiments.tabicl.tables.build_tabicl_train_llm_table --help
```

Use explicit input and output arguments for standalone runs: the historical
default paths refer to local experiment artifacts that are not distributed.
The online evaluator supplies its generated paths automatically. Reusable
TabICL implementations live in `experiments/modules/tabicl/`.
