"""Source-preserving GIN protocol: validation AUC selection and frozen negatives.

The model calculations retain the audited implementation. Entrypoint seed
defaults are reviewed separately and pinned with the source. This adapter
removes the historical trainer's tracking and per-epoch test calls, installs the
validation-AUC checkpoint rule, and records protocol checks locally.
"""
import ast
import hashlib
import json
from pathlib import Path

TRAINER = 'experiments/semantic_mlp/train_semantic_mlp_pipeline.py'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_sources(repo, pins_path):
    pins = json.loads(Path(pins_path).read_text(encoding='utf-8-sig'))
    mismatches = [name for name, digest in pins.items()
                  if not (repo / name).is_file() or sha256_file(repo / name) != digest]
    if mismatches:
        raise RuntimeError('Source differs from the reviewed version; review before repinning: '
                           + ', '.join(mismatches[:12]))
    return pins


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('Trainer source anchor changed: ' + old[:75])
    return source.replace(old, new, 1)


def adapt_source(source, report_test=True):
    """Select on validation AUC; optionally test the restored best checkpoint."""
    start = '    wandb = None\n'
    end = '    class DataArgs:\n'
    if source.count(start) != 1 or source.count(end) != 1:
        raise RuntimeError('Tracking initialization anchors changed')
    a, b = source.index(start), source.index(end)
    source = source[:a] + '    wandb = _gin_metrics\n\n' + source[b:]

    start = '        test_profile = {}\n'
    end = '        if args.profile_runtime:\n'
    if source.count(start) != 1:
        raise RuntimeError('Per-epoch test block changed')
    a = source.index(start)
    b = source.index(end, a)
    source = source[:a] + source[b:]
    lines = source.splitlines(keepends=True)
    test_log_lines = [line for line in lines if line.lstrip().startswith("'epoch_test/")]
    if len(test_log_lines) != 6:
        raise RuntimeError('Per-epoch test log schema changed')
    source = ''.join(line for line in lines if line not in test_log_lines)

    # Profiling blocks refer to the removed test profiles. The recipe has its own
    # validation-only timer, so remove these disabled upstream blocks entirely.
    tree = ast.parse(source)
    ranges = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Attribute)
                and isinstance(node.test.value, ast.Name)
                and node.test.value.id == 'args' and node.test.attr == 'profile_runtime'):
            ranges.append((node.lineno - 1, node.end_lineno))
    lines = source.splitlines(keepends=True)
    for first, last in sorted(ranges, reverse=True):
        del lines[first:last]
    source = ''.join(lines)

    tail = '    if not os.path.exists(args.checkpoint_path):\n'
    if source.count(tail) != 1:
        raise RuntimeError('Trainer final evaluation anchor changed')
    if report_test:
        # Preserve upstream restoration of every trained component and both
        # final test calls, then hand their metrics to our single local record.
        end = '    def _print_time_bucket_metrics(title: str, metrics: dict) -> None:\n'
        if source.count(end) != 1:
            raise RuntimeError('Trainer final test reporting anchor changed')
        source = source[:source.index(end)] + '    return _gin_finalize(args, test_metrics, new_node_metrics)\n'
        source = replace_once(source,
            "    ckpt = torch.load(args.checkpoint_path, map_location=device)\n",
            "    ckpt = torch.load(args.checkpoint_path, map_location=device)\n    _gin_begin_test(ckpt)\n")
    else:
        source = source[:source.index(tail)] + '    return _gin_finalize(args)\n'
    # The recipe objective, checkpoint choice and patience must all use AUC.
    # Keep the upstream file unchanged and require each reviewed anchor once.
    for old, new in [
        ('    best_val_ap = -1.0\n', '    best_val_auc = -1.0\n'),
        ("        if val_metrics['average_precision'] > (best_val_ap + args.early_stopping_min_delta):\n",
         "        if val_metrics['roc_auc'] > (best_val_auc + args.early_stopping_min_delta):\n"),
        ("            best_val_ap = val_metrics['average_precision']\n",
         "            best_val_auc = val_metrics['roc_auc']\n"),
        ("                'best_val_ap': best_val_ap,\n",
         "                'best_val_auc': best_val_auc,\n                'selection_metric': 'val/best_auc',\n                'selected_epoch': epoch,\n"),
        ("            'val/best_ap': best_val_ap,\n", "            'val/best_auc': best_val_auc,\n"),
        ('no val AP improvement for {epochs_no_improve} epochs',
         'no val AUC improvement for {epochs_no_improve} epochs'),
    ]:
        source = replace_once(source, old, new)
    source = source.replace('torch.load(args.checkpoint_path, map_location=device)',
                            'torch.load(args.checkpoint_path, map_location=device, weights_only=False)')
    compile(source, '<gin-trainer>', 'exec')
    return source


class FrozenNegativeSampler:
    """Replay canonical-256 split negatives independently of eval batching."""
    negative_sample_strategy = 'random'

    def __init__(self, sources, destinations):
        if len(sources) != len(destinations):
            raise ValueError('Negative table length mismatch')
        self.sources, self.destinations = sources, destinations
        self.position = 0

    @classmethod
    def build(cls, upstream_sampler, data, sample_fn, canonical_batch_size=256):
        import numpy as np
        if upstream_sampler.negative_sample_strategy != 'random':
            raise ValueError('This recipe pins random evaluation negatives')
        upstream_sampler.reset_random_state()
        sources, destinations = [], []
        for start in range(0, len(data.src_node_ids), canonical_batch_size):
            stop = start + canonical_batch_size
            src, dst = sample_fn(
                neg_sampler=upstream_sampler,
                src=data.src_node_ids[start:stop], dst=data.dst_node_ids[start:stop],
                times=data.node_interact_times[start:stop], num_negatives=1)
            sources.append(src)
            destinations.append(dst)
        upstream_sampler.reset_random_state()
        if not sources:
            raise ValueError('Evaluation split is empty')
        return cls(np.concatenate(sources), np.concatenate(destinations))

    def reset_random_state(self):
        self.position = 0

    def sample(self, size, **kwargs):
        stop = self.position + size
        if stop > len(self.destinations):
            raise RuntimeError('Negative replay exceeded the frozen query table')
        selected = slice(self.position, stop)
        self.position = stop
        return self.sources[selected], self.destinations[selected]

    def assert_complete(self):
        if self.position != len(self.destinations):
            raise RuntimeError('Evaluation did not consume every frozen negative')


def array_digest(*arrays):
    import numpy as np
    digest = hashlib.sha256()
    for values in arrays:
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


import math
import time

class TrialMetrics:
    def __init__(self, run):
        self.run = run
        self.validation = {}
        self.best = None
        self.evaluations = 0
        self.test_checkpoint_loaded = False
        self.test_evaluations = {}
        self.test_details = {}
        self.test_reported = False

    def log(self, payload, step=None):
        if any('test' in key.lower() for key in payload):
            raise RuntimeError('Test metrics must use the final checkpoint report')
        if self.test_checkpoint_loaded:
            raise RuntimeError('Training cannot resume after final test evaluation')
        ap = float(payload['val/ap'])
        auc = float(payload['val/auc'])
        if not math.isfinite(ap) or not math.isfinite(auc):
            raise RuntimeError('Nonfinite validation metric; fail this trial')
        if self.best is None or auc > self.best['auc']:
            self.best = {'ap': ap, 'auc': auc, 'epoch': int(step),
                         **self.validation}
        if not math.isclose(float(payload['val/best_auc']), self.best['auc'], abs_tol=1e-12):
            raise RuntimeError('Logged objective does not match AUC-selected checkpoint')
        self.run.log({**payload, **self.validation}, step=step, commit=True)
        self.run.summary.update({
            'selected/epoch': self.best['epoch'], 'selected/val_ap': self.best['ap'],
            'selected/val_auc': self.best['auc'],
        })

    def begin_test(self, checkpoint):
        if self.best is None or not self.evaluations or self.test_checkpoint_loaded:
            raise RuntimeError('Final test requires a validation-selected checkpoint exactly once')
        if (checkpoint.get('selection_metric') != 'val/best_auc'
                or checkpoint.get('selected_epoch') != self.best['epoch']
                or not math.isclose(float(checkpoint.get('best_val_auc', float('nan'))),
                                    self.best['auc'], rel_tol=0.0, abs_tol=1e-12)):
            raise RuntimeError('Restored checkpoint differs from the validation selection')
        self.test_checkpoint_loaded = True

    def log_test(self, transductive, inductive):
        if (not self.test_checkpoint_loaded or self.test_reported
                or self.test_evaluations != {'test': 1, 'test_inductive': 1}):
            raise RuntimeError('Final test requires exactly one evaluation of each test split')
        payload = {'test/selected_epoch': self.best['epoch'], **self.test_details}
        for split, result in [('transductive', transductive), ('inductive', inductive)]:
            for name, key in [('auc', 'roc_auc'), ('ap', 'average_precision'), ('mrr', 'mrr'),
                              ('auc_global', 'roc_auc_global'), ('ap_global', 'average_precision_global')]:
                value = float(result[key])
                if not math.isfinite(value):
                    raise RuntimeError('Nonfinite final test metric: ' + split + '/' + name)
                payload['test/' + split + '_' + name] = value
        payload['test/auc'] = payload['test/transductive_auc']
        self.run.log(payload, commit=True)
        self.run.summary.update({**payload, 'selected/test_auc': payload['test/auc'],
                                 'selected/test_inductive_auc': payload['test/inductive_auc']})
        self.test_reported = True
        return payload


def validation_evaluator(runtime, torch, metrics, trial_dir, report_test=False):
    original_eval = runtime.evaluate_split
    original_compute = runtime.compute_prediction_metrics
    tables = {}

    def evaluate(**kwargs):
        import numpy as np
        desc = kwargs.get('desc')
        if desc == 'Val':
            if metrics.test_checkpoint_loaded:
                raise RuntimeError('Validation cannot run after final test has begun')
            split, prefix = 'validation', 'val'
        elif report_test and desc in ('Test-Transductive', 'Test-Inductive'):
            if not metrics.test_checkpoint_loaded:
                raise RuntimeError('Test requires the restored validation-best checkpoint')
            split, prefix = ('test', 'test/transductive') if desc == 'Test-Transductive' else ('test_inductive', 'test/inductive')
            if metrics.test_evaluations.get(split, 0):
                raise RuntimeError('Test split may only be evaluated once per trial')
        else:
            raise RuntimeError('GIN evaluation is restricted to validation')
        if kwargs['num_negatives'] != 1 or kwargs['dtgb_eval_batch_size'] != 256:
            raise RuntimeError('Evaluation protocol changed')
        data = kwargs['data_source']
        digest = array_digest(data.src_node_ids, data.dst_node_ids,
                              data.node_interact_times, data.edge_ids)
        if split not in tables:
            replay = FrozenNegativeSampler.build(kwargs['neg_sampler'], data, runtime.sample_negatives)
            tables[split] = (replay, digest)
            neg_hash = array_digest(data.edge_ids, replay.sources, replay.destinations)
            np.savez(trial_dir / (split + '_negatives.npz'), query_edge_ids=data.edge_ids,
                     sources=replay.sources, destinations=replay.destinations)
            metrics.run.summary.update({'protocol/' + split + '_query_sha256': digest,
                                        'protocol/' + split + '_negative_sha256': neg_hash,
                                        'protocol/metric_batch_size': 256})
        replay, query_hash = tables[split]
        if digest != query_hash:
            raise RuntimeError('Evaluation population/order changed inside a trial')

        # Explicit ordered indices prevent a loader change from silently
        # assigning the replay table to different positive queries.
        batch_size = kwargs['data_loader'].batch_size
        kwargs['data_loader'] = [torch.arange(i, min(i + batch_size, len(data.src_node_ids)))
                                for i in range(0, len(data.src_node_ids), batch_size)]
        kwargs['neg_sampler'] = replay
        device = kwargs['embeddings'].device
        cuda = device.type == 'cuda'
        if cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        metric_seconds = 0.0

        def timed_compute(*args, **inner_kwargs):
            nonlocal metric_seconds
            start = time.perf_counter()
            try:
                return original_compute(*args, **inner_kwargs)
            finally:
                metric_seconds += time.perf_counter() - start

        runtime.compute_prediction_metrics = timed_compute
        started = time.perf_counter()
        try:
            with torch.no_grad():
                result = original_eval(**kwargs)
            if cuda:
                torch.cuda.synchronize(device)
        finally:
            runtime.compute_prediction_metrics = original_compute
        elapsed = time.perf_counter() - started
        replay.assert_complete()
        if result['dtgb_eval_batch_size'] != 256 or result['dtgb_metric_aggregation'] != 'batch_mean':
            raise RuntimeError('DTGB metric grouping did not remain fixed')
        prediction_seconds = max(elapsed - metric_seconds, 1e-12)
        details = {
            prefix + '/ap_global': result['average_precision_global'],
            prefix + '/auc_global': result['roc_auc_global'],
            prefix + '/evaluation_seconds': elapsed, prefix + '/metric_seconds': metric_seconds,
            prefix + '/prediction_seconds': prediction_seconds,
            prefix + '/positive_queries_per_second': len(data.src_node_ids) / prediction_seconds,
            prefix + '/peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20 if cuda else 0,
            prefix + '/peak_reserved_mib': torch.cuda.max_memory_reserved(device) / 2**20 if cuda else 0,
        }
        if split == 'validation':
            metrics.evaluations += 1
            metrics.validation = details
        else:
            metrics.test_evaluations[split] = 1
            metrics.test_details.update(details)
        return result
    return evaluate



SPLITS = {
    'Sample val negatives': 'validation',
    'Sample test negatives': 'test',
    'Sample inductive negatives': 'test_inductive',
}


def canonical_queries(runtime, record, **kwargs):
    import numpy as np
    if kwargs['num_negatives'] != 1:
        raise RuntimeError('GIN requires one fixed evaluation negative')
    data = kwargs['data_source']
    replay = FrozenNegativeSampler.build(kwargs['neg_sampler'], data, runtime.sample_negatives)
    split = SPLITS[kwargs['desc']]
    query = array_digest(data.src_node_ids, data.dst_node_ids, data.node_interact_times, data.edge_ids)
    negative = array_digest(data.edge_ids, replay.sources, replay.destinations)
    # Each dataset has its own canonical-256 table, rebuilt deterministically
    # from reset split samplers. Never compare another dataset to Enron hashes.
    fingerprints = {'query_sha256': query, 'negative_sha256': negative}
    if split in record and record[split] != fingerprints:
        raise RuntimeError('Heuristic precompute population or negatives changed: ' + split)
    record[split] = fingerprints
    return replay.sources, replay.destinations, np.asarray(data.node_interact_times, dtype=np.float64).copy()


def validate_fusion_checkpoint(checkpoint, mode, torch):
    """Validate the distinct learned components used by each fusion architecture."""
    meta = checkpoint['model_config']
    if mode not in ('residual', 'late_concat') or meta.get('semantic_aux_fusion_mode') != mode:
        raise RuntimeError('Checkpoint fusion mode differs from the trial configuration')
    if meta.get('semantic_project_dim') != 0 or checkpoint.get('semantic_projector_state_dict') is not None:
        raise RuntimeError('This recipe requires raw semantic embeddings without projection')
    if checkpoint.get('mplp_exact_fusion_state_dict') is not None:
        raise RuntimeError('MPLP must remain disabled')
    head = checkpoint.get('heuristic_fusion_state_dict')
    summary = {'heuristics/fusion_mode': mode}
    if mode == 'residual':
        if meta.get('auxiliary_dim') != 0 or head is None:
            raise RuntimeError('Residual fusion requires its separate linear head')
        weight, bias = head['linear.weight'], head['linear.bias']
        if tuple(weight.shape) != (1, 4) or tuple(bias.shape) != (1,) or not all(torch.isfinite(t).all() for t in (weight, bias)):
            raise RuntimeError('Invalid residual heuristic head')
        summary.update({'heuristics/selected_weights': dict(zip(('recency', 'popularity', 'past', 'ra'), weight.flatten().tolist())),
                        'heuristics/selected_bias': bias.item()})
    else:
        # late_concat learns auxiliary inputs inside the scorer, with no residual head.
        if meta.get('auxiliary_dim') != 4 or head is not None:
            raise RuntimeError('late_concat requires four MLP auxiliary inputs and no residual head')
        weight = checkpoint['state_dict']['mlp.0.weight']
        if weight.ndim != 2 or weight.shape[1] != 2 * meta['input_dim'] + 4 or not torch.isfinite(weight).all():
            raise RuntimeError('late_concat scorer does not contain the expected heuristic input columns')
        summary['heuristics/auxiliary_dim'] = 4
    return summary


def install_hooks(namespace, runtime, torch, metrics, trial_dir):
    import numpy as np
    started = time.perf_counter()
    events, epochs, negatives, restores = [], [], {}, []
    restored = {}

    def save():
        value = {'elapsed_total_seconds': time.perf_counter() - started,
                 'events': events, 'epochs': epochs, 'heuristic_negative_tables': negatives,
                 'restored_components': restores,
                 'timing_notes': 'Train function excludes heuristic precompute; evaluation uses cached heuristic features. Feature preparation is reported separately. Input hashing and final result/checkpoint hashing are excluded.'}
        path = trial_dir / 'timing.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2) + '\n')
        temp.replace(path)
        return value

    def timed(label, function, *args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        begin = time.perf_counter()
        value = function(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        events.append({'label': label, 'seconds': time.perf_counter() - begin,
                       'completed_epochs_before': metrics.evaluations})
        save()
        return value

    namespace['build_precomputed_negative_queries'] = lambda **kw: canonical_queries(runtime, negatives, **kw)
    original_train = namespace['train_one_epoch']
    namespace['train_one_epoch'] = lambda *a, **kw: timed('train_function', original_train, *a, **kw)
    original_pool = namespace['build_precomputed_train_negative_pool']
    namespace['build_precomputed_train_negative_pool'] = lambda *a, **kw: timed('train_negative_sampling', original_pool, *a, **kw)
    extractor_cls = namespace['HeuristicFeatureExtractor']
    original_precompute = extractor_cls.precompute_raw_features

    def precompute(self, *args, **kwargs):
        return timed(kwargs.get('desc', 'heuristic_precompute'), original_precompute, self, *args, **kwargs)
    extractor_cls.precompute_raw_features = precompute

    original_log = metrics.log
    def log(payload, step=None):
        pending = [e for e in events if e['completed_epochs_before'] == metrics.evaluations - 1]
        cost = {'cost/train_function_seconds': sum(e['seconds'] for e in pending if e['label'] == 'train_function'),
                'cost/train_heuristic_precompute_seconds': sum(e['seconds'] for e in pending if e['label'].startswith('Heuristics: train negatives')),
                'cost/train_negative_sampling_seconds': sum(e['seconds'] for e in pending if e['label'] == 'train_negative_sampling')}
        original_log({**payload, **cost}, step=step)
        epochs.append({'epoch': step, **payload, **metrics.validation, **cost})
        save()
    metrics.log = log

    original_begin_test = namespace['_gin_begin_test']
    def begin_test(checkpoint):
        original_begin_test(checkpoint)
        for key in ('state_dict', 'gcn_state_dict', 'semantic_projector_state_dict', 'heuristic_fusion_state_dict'):
            state = checkpoint.get(key)
            if state is not None:
                restored[key] = {name: tensor.detach().cpu().clone() for name, tensor in state.items()}
        summary = validate_fusion_checkpoint(checkpoint, metrics.run.config['semantic_aux_fusion_mode'], torch)
        metrics.run.summary.update(summary)
        save()
    namespace['_gin_begin_test'] = begin_test

    original_evaluate = namespace['evaluate_split']
    def evaluate(**kwargs):
        desc = kwargs['desc']
        if desc.startswith('Test-'):
            checked = []
            for name, key in (('model', 'state_dict'), ('gcn_encoder', 'gcn_state_dict'),
                              ('semantic_projector', 'semantic_projector_state_dict'),
                              ('heuristic_fusion', 'heuristic_fusion_state_dict')):
                module = kwargs.get(name)
                if module is None:
                    if key in restored:
                        raise RuntimeError('Missing restored component: ' + name)
                    continue
                actual = module.state_dict()
                expected = restored[key]
                if actual.keys() != expected.keys() or any(not torch.equal(t.detach().cpu(), expected[k]) for k, t in actual.items()):
                    raise RuntimeError('Component not restored for final test: ' + name)
                checked.append(name)
            restores.append({'split': desc, 'components': checked})
        # Check representative cached rows against their actual query triples.
        data = kwargs['data_source']
        split = {'Val': 'validation', 'Test-Transductive': 'test', 'Test-Inductive': 'test_inductive'}[desc]
        if array_digest(data.src_node_ids, data.dst_node_ids, data.node_interact_times, data.edge_ids) != negatives[split]['query_sha256']:
            raise RuntimeError('Evaluated queries do not match heuristic precompute')
        replay = FrozenNegativeSampler.build(kwargs['neg_sampler'], data, runtime.sample_negatives)
        if array_digest(data.edge_ids, replay.sources, replay.destinations) != negatives[split]['negative_sha256']:
            raise RuntimeError('Evaluated negatives do not match heuristic precompute')
        raw = kwargs['precomputed_neg_raw_heuristic_features']
        indices = np.unique(np.linspace(0, len(data.src_node_ids) - 1, 19, dtype=int))
        checked_raw = kwargs['heuristic_extractor'].get_raw_features(
            replay.sources[indices], replay.destinations[indices], data.node_interact_times[indices])
        if not np.array_equal(raw[indices], checked_raw) or not np.isfinite(raw).all():
            raise RuntimeError('Cached heuristic features mismatch evaluated queries')
        result = original_evaluate(**kwargs)
        save()
        return result
    namespace['evaluate_split'] = evaluate

    original_finalize = namespace['_gin_finalize']
    def finalize(*args, **kwargs):
        for split, item in negatives.items():
            if metrics.run.summary['protocol/' + split + '_negative_sha256'] != item['negative_sha256']:
                raise RuntimeError('Final negative table consistency failed')
        state = save()
        cost = {'cost/elapsed_before_finalization_seconds': state['elapsed_total_seconds'],
                'cost/train_function_total_seconds': sum(e['seconds'] for e in events if e['label'] == 'train_function'),
                'cost/train_heuristic_precompute_total_seconds': sum(e['seconds'] for e in events if e['label'].startswith('Heuristics: train negatives')),
                'cost/fixed_heuristic_precompute_total_seconds': sum(e['seconds'] for e in events if e['label'].startswith('Heuristics:') and not e['label'].startswith('Heuristics: train negatives')),
                'cost/train_negative_sampling_total_seconds': sum(e['seconds'] for e in events if e['label'] == 'train_negative_sampling'),
                'cost/validation_total_seconds': sum(e['val/evaluation_seconds'] for e in epochs),
                'cost/completed_epochs': len(epochs)}
        metrics.run.summary.update(cost)
        original_finalize(*args, **kwargs)
        path = trial_dir / 'result.json'
        result = json.loads(path.read_text())
        result['cost'] = cost
        result['heuristic_negative_tables'] = negatives
        result['restored_components'] = restores
        path.write_text(json.dumps(result, indent=2) + '\n')
    namespace['_gin_finalize'] = finalize
