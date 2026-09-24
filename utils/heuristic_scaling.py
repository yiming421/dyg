"""Training-fitted heuristic preprocessing shared by graph and sequence scorers."""
import numpy as np

PREPROCESSING_VERSION = 'elapsed-log-train-minmax-v1'


class FixedHeuristicScaling:
    def fit_normalization(self, training_features):
        """Fit once, using training candidates only."""
        if getattr(self, '_feature_min', None) is not None:
            raise RuntimeError('Heuristic normalization is already fitted.')
        values = np.asarray(training_features, dtype=np.float32)
        if (values.ndim != 2 or values.shape[1] != len(self.feature_names)
                or not len(values) or not np.isfinite(values).all()):
            raise ValueError('Expected finite, nonempty training heuristic features.')
        self._feature_min = values.min(axis=0, keepdims=True)
        self._feature_max = values.max(axis=0, keepdims=True)

    def normalize_raw_features(self, features):
        if getattr(self, '_feature_min', None) is None:
            raise RuntimeError('Fit heuristic normalization on training data or load its checkpoint state first.')
        values = np.asarray(features, dtype=np.float32)
        if (values.ndim != 2 or values.shape[1] != len(self.feature_names)
                or not np.isfinite(values).all()):
            raise ValueError('Invalid heuristic feature matrix.')
        denom = self._feature_max - self._feature_min
        denom = np.where(denom < 1e-12, 1.0, denom)
        return np.clip((values - self._feature_min) / denom, 0.0, 1.0).astype(np.float32)

    def normalization_state_dict(self):
        if getattr(self, '_feature_min', None) is None:
            raise RuntimeError('Cannot save unfitted heuristic normalization.')
        return {'version': PREPROCESSING_VERSION, 'features': list(self.feature_names),
                'minimum': self._feature_min.tolist(), 'maximum': self._feature_max.tolist()}

    def load_normalization_state_dict(self, state):
        if not isinstance(state, dict) or state.get('version') != PREPROCESSING_VERSION:
            raise ValueError('Missing or legacy heuristic preprocessing state; use its original code/checkpoint, or retrain with corrected preprocessing.')
        if state.get('features') != list(self.feature_names):
            raise ValueError('Checkpoint heuristic feature order does not match the extractor.')
        minimum = np.asarray(state.get('minimum'), dtype=np.float32)
        maximum = np.asarray(state.get('maximum'), dtype=np.float32)
        shape = (1, len(self.feature_names))
        if (minimum.shape != shape or maximum.shape != shape
                or not np.isfinite(minimum).all() or not np.isfinite(maximum).all()
                or np.any(maximum < minimum)):
            raise ValueError('Invalid checkpoint heuristic normalization bounds.')
        self._feature_min, self._feature_max = minimum.copy(), maximum.copy()


def fit_training_heuristic_normalization(extractor, sources, targets, times,
                                         positive_features=None, seed=42, chunk_size=200000):
    """Fit all training positives plus one frozen training-pool negative per query.

    Uses an independent RNG so fitting does not change training negative sampling.
    Chunking limits temporary memory; only the global training extrema are fitted.
    """
    sources = np.asarray(sources, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    times = np.asarray(times, dtype=np.float64)
    if (sources.ndim != 1 or sources.shape != targets.shape or sources.shape != times.shape
            or not len(sources) or chunk_size <= 0):
        raise ValueError('Training queries must be nonempty, aligned vectors.')
    if positive_features is not None and np.shape(positive_features) != (len(sources), len(extractor.feature_names)):
        raise ValueError('Precomputed training positives are not aligned.')
    rng = np.random.RandomState(seed)
    pool = np.unique(targets)
    bounds = []
    compute = getattr(extractor, '_compute_raw_features_batched', extractor.get_raw_features)
    for start in range(0, len(sources), chunk_size):
        stop = min(start + chunk_size, len(sources))
        src, dst, ts = sources[start:stop], targets[start:stop], times[start:stop]
        positive = (compute(src, dst, ts) if positive_features is None
                    else np.asarray(positive_features[start:stop], dtype=np.float32))
        negative = compute(src, rng.choice(pool, len(src)), ts)
        for values in (positive, negative):
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite training heuristic features.')
            bounds.extend([values.min(axis=0), values.max(axis=0)])
    extractor.fit_normalization(np.asarray(bounds, dtype=np.float32))
