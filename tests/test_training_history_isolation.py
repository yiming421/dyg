"""CPU regressions for graph isolation across training and deferred context."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.modules.llm_lp.training_protocol import (
    resolve_training_protocol, protocol_history_edges, tag_training_history,
    history_edges_for_samples, validate_protocol_samples,
)
from utils.graph_history import require_training_history_table, TRAIN_HISTORY_POLICY


class TrainingHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.tmp.name)
        cls.data = cls.base/'Enron'
        cls.data.mkdir()
        rng = np.random.RandomState(12)
        cls.edges = pd.DataFrame(dict(u=np.arange(600)%30,
            i=(np.arange(600)%30+rng.randint(1,30,600))%30,
            r=np.zeros(600,dtype=int), ts=np.arange(600)+1,
            label=np.ones(600,dtype=int)))
        cls.edges.to_csv(cls.data/'edge_list.csv',index=False)
        pd.DataFrame(dict(i=np.arange(30),text=[f'Entity {i}' for i in range(30)])).to_csv(cls.data/'entity_text.csv',index=False)
        cls.embedding=cls.base/'embedding.npy'
        np.save(cls.embedding,rng.randn(30,16).astype('float32'))
        cls.protocol=resolve_training_protocol(cls.edges,split_name='train')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_deferred_graph_resolution_and_table_provenance(self):
        samples=[dict(source_id=1,target_id=2,timestamp=400,label=1)]
        tag_training_history(samples,self.protocol)
        expected=protocol_history_edges(self.edges,self.protocol)
        actual=history_edges_for_samples(self.edges,samples)
        pd.testing.assert_frame_equal(actual,expected)
        pd.testing.assert_frame_equal(history_edges_for_samples(actual,samples),expected)
        bad=self.edges.copy();bad.loc[0,'i']=100
        with self.assertRaises(ValueError): history_edges_for_samples(bad,samples)
        with self.assertRaises(ValueError): require_training_history_table({})
        require_training_history_table({'train_history_policy':np.asarray(TRAIN_HISTORY_POLICY)})
        with self.assertRaises(ValueError):
            resolve_training_protocol(self.edges,split_name='train',train_data_protocol='legacy_time_only')
        with self.assertRaises(ValueError):
            resolve_training_protocol(self.edges,split_name='pretest')

    def capture_setup(self, scorer='mlp', mp='gin', rolling=True):
        from utils import DataLoader
        path=ROOT/'experiments/semantic_mlp/train_semantic_mlp_pipeline.py'
        source=path.read_text()
        anchor='    raw_input_dim = int(active_embeddings.shape[1])'
        self.assertEqual(source.count(anchor),1)
        source=source.replace(anchor,'    _capture(locals())\n'+anchor)
        class Captured(Exception): pass
        result={}
        def capture(values): result.update(values);raise Captured()
        ns={'__name__':'training_setup_regression','__file__':str(path),'_capture':capture}
        exec(compile(source,str(path),'exec'),ns)
        ns['launch_seed_workers']=lambda *a,**kw:False
        args=['trainer','--dataset_name','Enron','--gpu','-1','--seed','42',
              '--embedding_cache',str(self.embedding),'--entity_text_path',str(self.data/'entity_text.csv'),
              '--scorer_type',scorer,'--learnable_mp_type',mp,
              '--use_learnable_gcn',str(scorer=='mlp').lower(),
              '--use_heuristic_features','true','--use_gpu_heuristics','false',
              '--rolling_smoothing',str(rolling).lower(),'--train_holdout_recent_edges','7',
              '--smooth_time_window','1000000','--smooth_endpoint_topk_recent','0',
              '--ridge_projection_dim','0',
              '--train_batch_size','64','--eval_batch_size','64',
              '--checkpoint_path',str(self.base/'model.pt')]
        with patch.object(sys,'argv',args),patch.object(DataLoader,'_resolve_dataset_root',lambda _:str(self.data)),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(Captured): ns['main']()
        return result

    def test_tabicl_table_and_debug_history_contract(self):
        from experiments.modules.tabicl.online_pipeline import write_online_router_inputs
        from experiments.tabicl.tables.build_tabicl_train_llm_table import join_routed_debug
        support=[dict(query_id=i,source_id=1,target_id=2,timestamp=400+i,
                      label=1-i,semantic_mlp_score=.3+.2*i) for i in range(2)]
        deployment=[dict(s) for s in support]
        tag_training_history(support,self.protocol)
        table=self.base/'table.npz'; debug=self.base/'debug.jsonl'
        kwargs=dict(support_samples=support,support_llm_scores=np.array([.8,.2]),
                    deployment_samples=deployment,backbone_score_field='semantic_mlp_score',
                    route_center=.5,budget_fraction=.5,table_path=table,support_debug_path=debug)
        write_online_router_inputs(**kwargs)
        with np.load(table,allow_pickle=False) as archive:
            payload=dict(archive)
        require_training_history_table(payload)
        join_kwargs=dict(payload=payload,split='train',debug_jsonl=str(debug),
                         route_mask=np.ones(2,dtype=bool),expected_selected=2,
                         expected_debug_split='train',gnn_score_atol=1e-8)
        join_routed_debug(**join_kwargs)
        rows=[json.loads(line) for line in debug.read_text().splitlines()]
        for row in rows:row.pop('train_history_policy')
        debug.write_text('\n'.join(json.dumps(row) for row in rows))
        with self.assertRaises(ValueError):join_routed_debug(**join_kwargs)
        with self.assertRaises(ValueError):
            write_online_router_inputs(**dict(kwargs,support_samples=deployment))

    def test_gin_and_control_training_graphs(self):
        for scorer,mp,rolling in [('mlp','gin',True),('mlp','gcn',True),('mlp','attn_pool',True),
                ('cross_attention','gin',True),('dygformer_lite','gin',True),
                ('ncn','gin',True),('seqfilter','gin',True),('ridge','gin',True),
                ('mlp','gin',False)]:
            with self.subTest(scorer=scorer,mp=mp,rolling=rolling):
                ns=self.capture_setup(scorer,mp,rolling)
                train=ns['train_data'];full=ns['full_data']
                reserved=set(self.protocol.reserved_node_ids)
                self.assertFalse(reserved & set(train.src_node_ids))
                self.assertFalse(reserved & set(train.dst_node_ids))
                np.testing.assert_array_equal(ns['smooth_src'],train.src_node_ids)
                np.testing.assert_array_equal(ns['train_heuristic_extractor'].directed_src_node_ids,train.src_node_ids)
                np.testing.assert_array_equal(ns['heuristic_extractor'].directed_src_node_ids,full.src_node_ids)
                self.assertEqual(ns['train_heuristic_extractor'].normalization_state_dict(),ns['heuristic_extractor'].normalization_state_dict())
                for key in ('train_scorer_neighbor_index','train_mp_neighbor_index'):
                    index=ns[key]
                    if index is None:continue
                    for node in reserved:
                        self.assertEqual(len(index.node_neighbors[node]),0)
                    for neighbors in index.node_neighbors:
                        self.assertFalse(reserved & set(neighbors))
                provider=ns['train_rolling_provider']
                if provider is not None:
                    np.testing.assert_array_equal(provider.init_src_node_ids,train.src_node_ids)
                    np.testing.assert_array_equal(provider.init_node_interact_times,train.node_interact_times)
                if mp=='gin' and scorer=='mlp' and rolling:
                    import torch
                    provider.prepare_batch(np.array([400.]))
                    adj=provider.current_sum_adj.coalesce()
                    off_diagonal=adj.indices()[0]!=adj.indices()[1]
                    involved=set(adj.indices()[:,off_diagonal].flatten().tolist())
                    self.assertFalse(reserved & involved)
                    from experiments.modules.semantic_mlp.graph_components import (
                        LearnableGINEncoder, LearnableGCNEncoder, apply_static_smoothing_operator,
                    )
                    from experiments.modules.heuristic_semantic_models import smooth_embeddings_by_time_window_torch
                    _,norm_adj=smooth_embeddings_by_time_window_torch(
                        embeddings=np.load(self.embedding),src_node_ids=train.src_node_ids,
                        dst_node_ids=train.dst_node_ids,node_interact_times=train.node_interact_times,
                        time_window=1000000,reference_time=399.5,device='cpu',
                        supernode_strength=.5,return_norm_adj=True)
                    for encoder,operator in (
                        (LearnableGINEncoder(16,16,num_layers=2),adj),
                        (LearnableGCNEncoder(16,16,num_layers=2),norm_adj),
                        (lambda x,a:apply_static_smoothing_operator(x,a,2),norm_adj),
                    ):
                        x=torch.from_numpy(np.load(self.embedding)).requires_grad_()
                        out=encoder(x,operator)
                        out[int(train.src_node_ids[0]),0].backward()
                        self.assertEqual(torch.count_nonzero(x.grad[list(reserved)]).item(),0)
                    evaluation=ns['val_rolling_provider'];evaluation.prepare_batch(np.array([500.]))
                    eval_edges=evaluation.current_sum_adj.coalesce().indices()
                    self.assertTrue(reserved & set(eval_edges[:,eval_edges[0]!=eval_edges[1]].flatten().tolist()))

    def test_llm_training_eager_deferred_and_peft_defaults(self):
        from experiments.modules.llm_lp.sample_builder import create_test_samples
        from experiments.modules.llm_lp.prompt_context import materialize_samples_prompt_context
        from experiments.modules.llm_lp.peft import create_direct_peft_samples
        entities={i:str(i) for i in range(30)}
        common=dict(num_samples=12,history_window=47,random_seed=42,
                    compute_expert_prediction=False,compute_rrf_scores=False,
                    skip_key_signal_calibration=True,apply_gdelt_time_bucket=False)
        with contextlib.redirect_stdout(io.StringIO()):
            eager=create_test_samples(self.edges,entities,{0:'edge'},eval_split='train',
                defer_prompt_context_materialization=False,**common)
            deferred=create_test_samples(self.edges,entities,{0:'edge'},eval_split='train',
                defer_prompt_context_materialization=True,defer_postprocessing=True,**common)
            # Supplying the full stream again must not broaden the train graph.
            materialize_samples_prompt_context(deferred,edges_df=self.edges,
                entity_map=entities,history_window=47,use_gpu_heuristics=False)
            peft=create_direct_peft_samples(self.edges,entities,{0:'edge'},split_name='train',
                negative_ratio=1,defer_prompt_context_materialization=False,**common)
        for samples in (eager,deferred,peft):
            self.assertTrue(all(s['graph_history_scope']=='train' for s in samples))
            validate_protocol_samples(samples,self.protocol)
            history=protocol_history_edges(self.edges,self.protocol)
            events=list(history.sort_values('ts',kind='stable')[['u','r','i','ts']].itertuples(index=False,name=None))
            for sample in samples:
                for field,node in [('source_history',sample['source_id']),('target_history',sample['target_id'])]:
                    expected=[e for e in events if e[3]<sample['timestamp']
                              for endpoint in (e[0],e[2]) if endpoint==node][-47:]
                    self.assertEqual(sample[field],expected)


if __name__=='__main__': unittest.main(verbosity=2)
