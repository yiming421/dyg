"""CPU integration regression for the public LLM history protocol.

Run from the repository root with the normal LLM dependencies installed:
python tests/test_llm_history_integration.py
"""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from experiments.modules.llm_lp.prompt_context import (
    materialize_samples_prompt_context, calibrate_prompt_key_signals,
)
from experiments.modules.llm_lp.sample_builder import create_test_samples


class HistoryIntegrationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(12)
        self.df = pd.DataFrame(dict(u=rng.randint(1, 9, 120),
            i=rng.randint(1, 9, 120), ts=np.arange(120)//3, r=np.arange(120)))
        self.entities = {i: str(i) for i in range(1, 11)}
        self.relations = {i: str(i) for i in range(120)}

    def check_oracle(self, samples, cap):
        events = list(self.df.sort_values('ts', kind='stable')[['u','r','i','ts']]
                      .itertuples(index=False, name=None))
        for s in samples:
            u, v, t = s['source_id'], s['target_id'], s['timestamp']
            for field, node in [('source_history', u), ('target_history', v)]:
                expected = [e for e in events if e[3] < t
                            for endpoint in (e[0], e[2]) if endpoint == node][-cap:]
                self.assertEqual(s[field], expected)
            expected_count = sum(e[0] == u and e[2] == v and e[3] < t for e in events)
            self.assertEqual(s.get('num_past_interactions_raw', s['num_past_interactions']),
                             expected_count)

    def test_cached_endpoints_direction_and_cutoff(self):
        for cap in (1, 47, 50):
            samples = [dict(source_id=u, target_id=v, timestamp=t, label=1, query_id=j)
                       for j, (u,v,t) in enumerate([(1,2,0), (1,2,20), (1,3,20),
                           (2,1,40), (9,1,40), (1,9,40), (1,1,40)])]
            materialize_samples_prompt_context(samples, edges_df=self.df,
                entity_map=self.entities, history_window=cap, use_gpu_heuristics=False)
            self.check_oracle(samples, cap)

    def test_eager_deferred_positive_negative_equivalence(self):
        common = dict(num_samples=12, history_window=47, random_seed=42,
            compute_expert_prediction=False, compute_rrf_scores=False,
            skip_key_signal_calibration=True, defer_postprocessing=True,
            apply_gdelt_time_bucket=False)
        eager = create_test_samples(self.df, self.entities, self.relations,
                                   defer_prompt_context_materialization=False, **common)
        deferred = create_test_samples(self.df, self.entities, self.relations,
                                      defer_prompt_context_materialization=True, **common)
        materialize_samples_prompt_context(deferred, edges_df=self.df,
            entity_map=self.entities, history_window=47, use_gpu_heuristics=False)
        self.check_oracle(eager, 47)
        self.check_oracle(deferred, 47)
        self.assertEqual({s['label'] for s in eager}, {0, 1})
        keys = ('source_id','target_id','timestamp','source_history','target_history',
                'num_past_interactions_raw')
        self.assertEqual([[s[k] for k in keys] for s in eager],
                         [[s[k] for k in keys] for s in deferred])

    def test_reverse_only_count_zero_and_calibration(self):
        # Incoming events must be visible in endpoint history, but must not
        # inflate either the raw source->target count or its reference counts.
        base = pd.DataFrame([(1,0,2,1), (1,1,3,2), (1,2,3,3)],
                            columns=['u','r','i','ts'])
        with_reverse = pd.concat([base, pd.DataFrame([(2,3,1,4), (4,4,1,5)],
                                    columns=base.columns)], ignore_index=True)
        outputs = []
        for df in (base, with_reverse):
            samples = [dict(source_id=1, target_id=v, timestamp=10, label=1, query_id=j)
                       for j,v in enumerate((2,3,4))]
            materialize_samples_prompt_context(samples, edges_df=df,
                entity_map=self.entities, history_window=47, use_gpu_heuristics=False)
            self.assertEqual([s['num_past_interactions_raw'] for s in samples], [1,2,0])
            calibrate_prompt_key_signals(samples, edges_df=df,
                                        key_signal_reference='contextual',
                                        use_gpu_heuristics=False)
            outputs.append([s['num_past_interactions_pct'] for s in samples])
        self.assertEqual(outputs[0], outputs[1])
        self.assertIn((4,4,1,5), samples[2]['source_history'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
