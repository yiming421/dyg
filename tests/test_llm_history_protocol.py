import ast
from bisect import bisect_left
from collections import defaultdict
import importlib.util
from pathlib import Path
import random
import unittest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / 'experiments/modules/llm_lp/sample_builder.py'

def retrieval(events, cap):
    tree = ast.parse(BUILDER.read_text(encoding='utf-8'))
    create = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'create_test_samples')
    names = {'get_recent_history', 'build_source_history', 'build_target_history'}
    body = [n for n in create.body if isinstance(n, ast.FunctionDef) and n.name in names]
    env = dict(bisect_left=bisect_left, semantic_history=False, history_window=cap,
               history_as_endpoint=defaultdict(list))
    # Execute the production indexing block, without importing GPU dependencies.
    indexing = next(n for n in create.body if isinstance(n, ast.If) and
                    isinstance(n.test, ast.Name) and n.test.id == 'need_event_histories' and
                    any(isinstance(x, ast.For) for x in n.body))
    env.update(need_event_histories=True, history_as_source=defaultdict(list),
               history_as_target=defaultdict(list), pair_history_as_source=defaultdict(lambda: defaultdict(list)),
               tqdm=lambda it, **kw: it)
    ordered = sorted(events, key=lambda e: e[3])
    env.update(dict(zip(['u_vals','r_vals','i_vals','ts_vals'], zip(*ordered))))
    exec(compile(ast.Module(body=[indexing]+body, type_ignores=[]), str(BUILDER), 'exec'), env)
    return env['build_source_history'], env['build_target_history']

class HistoryTests(unittest.TestCase):
    def test_graph_oracle_and_endpoint_symmetry(self):
        rng = random.Random(12)
        events = [(rng.randrange(5), i, rng.randrange(5), rng.randrange(20)) for i in range(130)]
        # Repeated events remain separate; self loops follow graph's two adjacency entries.
        events += [events[0], events[0]]
        for cap in (1, 3, 47, 50):
            source, target = retrieval(events, cap)
            for node in range(6):
                for time in (0, 5, 19, 20):
                    oracle = []
                    for e in sorted(events, key=lambda e: e[3]):
                        if e[3] >= time: continue
                        if e[0] == node: oracle.append(e)
                        if e[2] == node: oracle.append(e)
                    expected = oracle[-cap:]
                    self.assertEqual(source(node, 99, time)[0], expected)
                    self.assertEqual(target(99, node, time)[0], expected)

    def test_prompt_preserves_incoming_direction(self):
        path = ROOT / 'experiments/modules/llm_lp/prompt_template.py'
        spec = importlib.util.spec_from_file_location('aligned_prompt', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        fn = module.create_prompt
        prompt = fn(relation='contacts', source_entity='Alice', target_entity='Bob', source_id=1, relation_id=1,
                    target_id=2, prediction_time=10, source_history=[(3,1,1,5)], target_history=[],
                    entity_map={1:'Alice',2:'Bob',3:'Carol'}, relation_map={1:'contacts'},
                    use_chat_template=False, use_cot=False, include_key_signals=False,
                    include_expert_prediction=False, ablate_mutual_history=True,
                    ablate_common_neighbors=True)
        self.assertIn('up to 47 events', prompt)
        self.assertIn('received BY Alice or performed BY them', prompt)
        self.assertIn('Carol --> Alice', prompt)
        self.assertNotIn('initiated BY Alice', prompt)

    def test_entrypoint_defaults_and_overrides(self):
        import argparse
        for rel in ['experiments/modules/llm_lp/cli.py','experiments/llm_lp/train_peft_link_prediction.py']:
            tree = ast.parse((ROOT/rel).read_text(encoding='utf-8'))
            call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and
                        isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument' and
                        n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == '--history_window')
            env = dict(parser=argparse.ArgumentParser())
            exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=call)], type_ignores=[])), rel, 'exec'), env)
            self.assertEqual(env['parser'].parse_args([]).history_window, 47)
            self.assertEqual(env['parser'].parse_args(['--history_window','9']).history_window, 9)

if __name__ == '__main__': unittest.main(verbosity=2)
