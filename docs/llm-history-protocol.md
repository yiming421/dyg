# LLM history and interaction counts

For each query `(source, target, t)`, both endpoint histories include incoming
and outgoing events strictly before `t`, merged in stable chronological order.
The most recent `K` events are retained per endpoint; the default is `K=47`
and `--history_window` overrides it. Original event arrows and repeated events
are retained. Self-loops follow the graph sampler's two endpoint entries.
The same rule applies to eager construction and deferred/cached materialization.

The past-interaction count is directed: only `source -> target` events before
`t` contribute, across all prior history rather than just the prompt window.
Negative candidates use the same rule. Contextual count calibration uses the
source's outgoing counterpart counts. The mutual-history section and pair
recency continue to use both directions; these are separate from the directed
count feature.

Run the CPU regressions with:

```bash
python tests/test_llm_history_protocol.py
python tests/test_llm_history_integration.py
```

The September 24 release imports the earlier endpoint-history correction and
also corrects the directed count and its calibration. New runs record
`history_protocol=both_endpoints_recent_v1` and
`interaction_count_direction=source_to_target`. Regenerate prompt caches for
this protocol. Results and checkpoints produced by the previous asymmetric
history or bidirectional-count implementation are historical results, not
measurements of this corrected protocol. Use a new output directory; strict
training resumption checks these protocol fields and the history window.
