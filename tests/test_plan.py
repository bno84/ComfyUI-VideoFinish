"""Which post-process steps run, and their sizes; Lanczos coefficients. Pure Python plus torch."""

import unittest

import torch

from helpers import load

P = load("plan")
Z = load("resize")


def steps(w, h, upscale=None, fit="off", size=1080, interpolate="none"):
    plan = P.make(upscale is not None, "model.pth", fit, size, interpolate, "rife.pkl")
    return P.resolve(plan, w, h, upscale)


def size(w, h, **kwargs):
    return P.output_size(steps(w, h, **kwargs)[0], w, h)


class PlanTest(unittest.TestCase):
    def test_defaults_do_nothing(self):
        plan = P.make()
        self.assertFalse(P.enabled(plan))
        resolved, notes = P.resolve(plan, 864, 480)
        self.assertEqual(resolved, [])
        self.assertEqual(notes, ["upscale off", "resize off", "interpolation off"])

    def test_upscale_then_resize(self):
        self.assertEqual(size(1024, 576, upscale=2, fit="width", size=1920), (1920, 1080))
        self.assertEqual(size(1024, 576, upscale=2, fit="height", size=1080), (1920, 1080))

    def test_inferred_side_rounds_to_even(self):
        resolved, _ = steps(864, 480, upscale=4, fit="width", size=1920)
        self.assertEqual([s["op"] for s in resolved], ["upscale", "resize"])
        self.assertEqual(resolved[0]["size"], (3456, 1920))
        self.assertEqual(resolved[-1]["size"], (1920, 1066))     # 1066.67 → nearest even

    def test_resize_skipped_when_already_that_size(self):
        resolved, notes = steps(864, 480, upscale=2, fit="height", size=960)
        self.assertEqual([s["op"] for s in resolved], ["upscale"])
        self.assertIn("resize skipped: already 1728x960", notes)

    def test_resize_without_upscale(self):
        resolved, _ = steps(1920, 1080, fit="height", size=720)
        self.assertEqual(resolved, [{"op": "resize", "size": (1280, 720)}])
        self.assertEqual(size(864, 480, fit="height", size=1080), (1944, 1080))

    def test_interpolation_alone_and_combined(self):
        resolved, _ = steps(864, 480, interpolate="3x")
        self.assertEqual([s["op"] for s in resolved], ["interpolate"])
        self.assertEqual(P.multiplier(resolved), 3)
        self.assertEqual(P.output_size(resolved, 864, 480), (864, 480))
        resolved, _ = steps(864, 480, upscale=4, fit="height", size=1080, interpolate="4x")
        self.assertEqual([s["op"] for s in resolved], ["upscale", "resize", "interpolate"])
        self.assertIn("96 fps", P.describe(resolved, 864, 480, 24))

    def test_odd_results_are_refused(self):
        with self.assertRaisesRegex(ValueError, "even sizes"):
            steps(865, 480, interpolate="2x")

    def test_upscale_needs_the_model_scale(self):
        with self.assertRaisesRegex(ValueError, "scale"):
            P.resolve(P.make(upscale=True, upscale_model="m"), 64, 64)

    def test_enabled_steps_need_a_model(self):
        with self.assertRaisesRegex(ValueError, "upscale model"):
            P.make(upscale=True, upscale_model=P.NONE)
        with self.assertRaisesRegex(ValueError, "RIFE model"):
            P.make(interpolate="2x", rife_model=P.NONE)
        self.assertTrue(P.enabled(P.make(fit="height", size=720, upscale_model=P.NONE, rife_model=P.NONE)))

    def test_unknown_choices(self):
        with self.assertRaises(ValueError):
            P.make(fit="keep upscaled size")
        with self.assertRaises(ValueError):
            P.make(interpolate="5x")


class LanczosTest(unittest.TestCase):
    def test_same_size_is_identity(self):
        for i, (lo, weights) in enumerate(Z.lanczos_taps(10, 10)):
            for j, w in enumerate(weights, lo):
                self.assertAlmostEqual(w, 1.0 if j == i else 0.0, places=12)

    def test_weights_sum_to_one_and_widen_when_shrinking(self):
        for n_in, n_out in ((3456, 1944), (1728, 1944), (1920, 1080), (5, 3)):
            taps = Z.lanczos_taps(n_in, n_out)
            self.assertEqual(len(taps), n_out)
            for lo, weights in taps:
                self.assertAlmostEqual(sum(weights), 1.0, places=9)
                self.assertTrue(0 <= lo and lo + len(weights) <= n_in)
        # shrinking by 1.78x widens lanczos3's ±3 support to ±5.33: 11 taps; enlarging keeps 6
        self.assertEqual(max(len(w) for _, w in Z.lanczos_taps(3456, 1944)), 11)
        self.assertEqual(max(len(w) for _, w in Z.lanczos_taps(1728, 1944)), 6)

    def test_matches_pil_within_rounding(self):
        from PIL import Image
        import numpy as np
        y, x = np.mgrid[0:37, 0:53]
        image = np.stack([128 + 100 * np.sin(x / 5 + c) * np.cos(y / 7) for c in range(3)], -1).round().astype(np.uint8)
        for size in ((31, 20), (106, 74)):
            ours = Z.Resizer(*size)(torch.from_numpy(image).permute(2, 0, 1)[None].float() / 255)
            ours = (ours[0].permute(1, 2, 0) * 255).numpy()
            pil = np.asarray(Image.fromarray(image).resize(size, Image.Resampling.LANCZOS)).astype(np.float32)
            # PIL rounds to 8 bits between its passes; ours stays float.
            self.assertLess(np.abs(ours - pil).mean(), 0.5)
            self.assertLessEqual(np.abs(ours - pil).max(), 1.5)

    def test_keeps_float_precision(self):
        ramp = (torch.arange(64).float() / (4 * 255) + 0.25).view(1, 1, 1, 64).expand(1, 3, 8, 64)
        out = Z.Resizer(128, 16)(ramp)
        self.assertEqual(out.dtype, torch.float32)
        off_grid = ((out * 255) - (out * 255).round()).abs() > 0.05
        self.assertGreater(off_grid.float().mean().item(), 0.5)


if __name__ == "__main__":
    unittest.main()
