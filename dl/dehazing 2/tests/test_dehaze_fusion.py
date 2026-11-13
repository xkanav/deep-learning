import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dehaze_fusion.model import Gate, objective, dark_channel, n_features
from dehaze_fusion.data import load, check_splits
from dehaze_fusion.metrics import metrics, ssim_map, evaluate
from pack_sample import pack


def sample(k=2, h=12, w=10):
    rng = np.random.default_rng(8)
    return dict(hazy=rng.random((3, h, w)), target=rng.random((3, h, w)),
                predictions=rng.random((k, 3, h, w)), models=np.array([f'm{i}' for i in range(k)]),
                valid=np.ones((h, w)), scene=np.array('scene'))


class Tests(unittest.TestCase):
    def test_gradients(self):
        for k in (2, 3):
            d = sample(k)
            d['valid'][0, 0] = 0
            g = Gate(d['models'])
            g.w = np.random.default_rng(1).normal(0, .5, g.w.shape)
            initial = g.w.copy()
            g.step(d, lr=1, optimizer='sgd')
            analytic = initial - g.w
            for idx in [(0, 0), (1, 1), (n_features(k) - 1, k - 1)]:
                g.w = initial.copy()
                g.w[idx] += 1e-5
                plus = objective(g.forward(d)[0], d['target'], d['valid'], .1)[0]
                g.w[idx] -= 2e-5
                minus = objective(g.forward(d)[0], d['target'], d['valid'], .1)[0]
                self.assertAlmostEqual(analytic[idx], (plus - minus) / 2e-5, places=7)

    def test_convex_mixture(self):
        for k in (2, 3):
            d = sample(k)
            out, w, _ = Gate(d['models']).forward(d)
            np.testing.assert_allclose(w.sum(axis=-1), 1)
            np.testing.assert_allclose(out, d['predictions'].mean(axis=0))  # zero weights -> plain average
            self.assertTrue(((out >= d['predictions'].min(0) - 1e-12) & (out <= d['predictions'].max(0) + 1e-12)).all())

    def test_dark_channel(self):
        img = np.ones((3, 9, 9))
        img[1, 4, 4] = 0
        dc = dark_channel(img, r=1)
        self.assertEqual(dc[4, 4], 0)
        self.assertEqual(dc[3, 3], 0)
        self.assertEqual(dc[0, 0], 1)

    def test_checkpoint_and_no_target(self):
        d = sample()
        with tempfile.TemporaryDirectory() as t:
            g = Gate(d['models'])
            g.step(d)
            path = Path(t) / 'gate.npz'
            g.save(path)
            np.testing.assert_array_equal(g.forward(d)[0], Gate.restore(path).forward(d)[0])
            del d['target']
            np.savez(Path(t) / 's.npz', **d)
            load(Path(t) / 's.npz', target=False)

    def test_validation_and_leakage(self):
        with tempfile.TemporaryDirectory() as t:
            for split in ['train', 'val', 'test']:
                (Path(t) / split).mkdir()
                np.savez(Path(t) / split / 'x.npz', **sample())
            with self.assertRaisesRegex(ValueError, 'leakage'):
                check_splits(t)
            d = sample()
            d['predictions'][0, 0, 0, 0] = np.nan
            np.savez(Path(t) / 'bad.npz', **d)
            with self.assertRaises(ValueError):
                load(Path(t) / 'bad.npz')
            d = sample()
            d['hazy'] = d['hazy'][:2]
            np.savez(Path(t) / 'bad2.npz', **d)
            with self.assertRaisesRegex(ValueError, 'RGB'):
                load(Path(t) / 'bad2.npz')

    def test_metrics(self):
        d = sample(h=20, w=20)
        s = ssim_map(d['target'], d['target'])
        np.testing.assert_allclose(s, 1)
        r = metrics(d['target'], d['target'], d['valid'], s)
        self.assertEqual(r['mae'], 0)
        self.assertTrue(r['perfect_match'])
        self.assertAlmostEqual(r['ssim'], 1)
        self.assertIsNone(metrics(d['target'], d['target'], np.zeros((20, 20))))
        noisy = np.clip(d['target'] + .1, 0, 1)
        self.assertLess(ssim_map(noisy, d['target']).mean(), 1)

    def test_pack_png_and_evaluate(self):
        from PIL import Image
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            names = {}
            for n in ['hazy', 'clear', 'a', 'b']:
                names[n] = t / f'{n}.png'
                Image.fromarray((rng.random((16, 16, 3)) * 255).astype('uint8')).save(names[n])
            out = pack(names['hazy'], [names['a'], names['b']], ['A', 'B'], 'loc1', t / 'd' / 's.npz', names['clear'])
            d = load(out)
            self.assertEqual(d['hazy'].shape, (3, 16, 16))
            rep = evaluate(Gate(['A', 'B']), [out])
            methods = {r['method'] for r in rep['summary']}
            self.assertEqual(methods, {'hazy_input', 'mean_raw', 'gated', 'A', 'B'})


if __name__ == '__main__':
    unittest.main()
