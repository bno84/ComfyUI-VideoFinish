"""The vendored RIFE 4.26 network.

Opt-in checks against real files (skipped otherwise):
  VIDEOFINISH_RIFE_CHECKPOINT=/path/rife4.26.pkl  loads Practical-RIFE's flownet.pkl with strict=True.
  VIDEOFINISH_RIFE_REFERENCE=/path/rife_arch.py   compares outputs with ComfyUI-Frame-Interpolation's
                                                  rife_arch.py (needs ComfyUI on PYTHONPATH).
"""

import importlib.util
import os
import sys
import unittest

import torch

from helpers import load

ifnet = load("rife.ifnet")


class IFNetTest(unittest.TestCase):
    def test_shapes_and_endpoints(self):
        torch.manual_seed(0)
        model = ifnet.IFNet().eval()
        a, b = torch.rand(1, 3, 40, 72), torch.rand(1, 3, 40, 72)
        with torch.inference_mode():
            out = model(a, b, 0.5)
        self.assertEqual(out.shape, (1, 3, 40, 72))
        self.assertTrue(torch.isfinite(out).all())

    @unittest.skipUnless(os.environ.get("VIDEOFINISH_RIFE_CHECKPOINT"), "set VIDEOFINISH_RIFE_CHECKPOINT")
    def test_published_checkpoint_loads_strictly(self):
        state = torch.load(os.environ["VIDEOFINISH_RIFE_CHECKPOINT"], map_location="cpu", weights_only=True)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        state = {k: v for k, v in state.items() if not k.startswith(("teacher.", "caltime."))}
        model = ifnet.IFNet()
        model.load_state_dict(state, strict=True)
        # A half-way frame between two shifted copies of a smooth image lies between them.
        y, x = torch.meshgrid(torch.linspace(0, 6, 64), torch.linspace(0, 6, 96), indexing="ij")
        image = torch.stack([(torch.sin(x + c) * torch.cos(y) + 1) / 2 for c in range(3)])[None]
        a, b = image[..., :, :-4], image[..., :, 4:]
        with torch.inference_mode():
            middle = model.eval()(a, b, 0.5)
        target = image[..., :, 2:-2]
        self.assertLess((middle - target)[..., 8:-8, 8:-8].abs().mean().item(), 0.02)

    @unittest.skipUnless(os.environ.get("VIDEOFINISH_RIFE_REFERENCE"), "set VIDEOFINISH_RIFE_REFERENCE")
    def test_matches_the_reference_architecture(self):
        spec = importlib.util.spec_from_file_location("rife_reference", os.environ["VIDEOFINISH_RIFE_REFERENCE"])
        reference = importlib.util.module_from_spec(spec)
        argv, sys.argv = sys.argv, [sys.argv[0], "--cpu"]
        try:
            import comfy.options
            comfy.options.enable_args_parsing()
            spec.loader.exec_module(reference)
        finally:
            sys.argv = argv
        torch.manual_seed(1)
        ours = ifnet.IFNet().eval()
        theirs = reference.IFNet(arch_ver="4.26").eval()
        theirs.load_state_dict(ours.state_dict(), strict=True)
        a, b = torch.rand(1, 3, 72, 128), torch.rand(1, 3, 72, 128)
        with torch.inference_mode():
            for t in (0.25, 0.5, 0.75):
                expected = theirs(a, b, timestep=t, training=False, fastmode=True, ensemble=False)
                torch.testing.assert_close(ours(a, b, t), expected, rtol=0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
