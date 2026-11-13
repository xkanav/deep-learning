import tempfile
import unittest
from pathlib import Path
import numpy as np
from fusion.model import Gate, objective
from fusion.data import load, check_splits
from fusion.metrics import metrics


def sample(k=3):
    rng = np.random.default_rng(8)
    return dict(cloudy=rng.random((13, 4, 5)), target=rng.random((13, 4, 5)),
                predictions=rng.random((k, 13, 4, 5)), models=np.array([f'm{i}' for i in range(k)]),
                cloud_mask=rng.random((4, 5)), valid=np.ones((4, 5)), scene=np.array('scene'))


class Tests(unittest.TestCase):
    def test_gradients(self):
        d = sample()
        d['valid'][0, 0] = 0
        g = Gate(d['models'])
        initial = g.w.copy()
        g.step(d, lr=1)
        analytic = initial - g.w
        for idx in [(0, 0), (1, 2), (6, 1)]:
            g.w = initial.copy()
            g.w[idx] += 1e-5
            plus = objective(g.forward(d)[0], d['target'], d['valid'], .1)[0]
            g.w[idx] -= 2e-5
            minus = objective(g.forward(d)[0], d['target'], d['valid'], .1)[0]
            self.assertAlmostEqual(analytic[idx], (plus-minus)/2e-5, places=8)

    def test_two_three_and_clear_identity(self):
        for k in (2, 3):
            d = sample(k)
            d['cloud_mask'][:] = 0
            out, w, _ = Gate(d['models']).forward(d)
            np.testing.assert_array_equal(out, d['cloudy'])
            np.testing.assert_allclose(w.sum(axis=-1), 1)

    def test_checkpoint_and_no_target(self):
        d = sample()
        with tempfile.TemporaryDirectory() as t:
            g = Gate(d['models'])
            g.step(d)
            path = Path(t) / 'gate.npz'
            g.save(path)
            np.testing.assert_array_equal(g.forward(d)[0], Gate.restore(path).forward(d)[0])
            del d['target']
            path = Path(t) / 'sample.npz'
            np.savez(path, **d)
            load(path, target=False)

    def test_validation_and_leakage(self):
        with tempfile.TemporaryDirectory() as t:
            for split in ['train', 'val', 'test']:
                dest = Path(t) / split
                dest.mkdir()
                np.savez(dest / 'x.npz', **sample())
            with self.assertRaisesRegex(ValueError, 'leakage'):
                check_splits(t)
            d = sample()
            d['predictions'][0, 0, 0, 0] = np.nan
            np.savez(Path(t) / 'bad.npz', **d)
            with self.assertRaises(ValueError):
                load(Path(t) / 'bad.npz')

    def test_metrics(self):
        d = sample()
        r = metrics(d['target'], d['target'], d['valid'])
        self.assertEqual(r['mae'], 0)
        self.assertTrue(r['perfect_match'])
        self.assertIsNone(metrics(d['target'], d['target'], np.zeros((4, 5))))


if __name__ == '__main__':
    unittest.main()
