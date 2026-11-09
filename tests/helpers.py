"""Shared test helpers. The pack is imported as package `videofinish` without running its __init__,
so the encoder and plan tests need only PyAV and torch, not ComfyUI."""

import importlib
import math
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import patch

import av
import numpy as np
import torch

PACK = Path(__file__).resolve().parent.parent

if "videofinish" not in sys.modules:
    package = types.ModuleType("videofinish")
    package.__path__ = [str(PACK)]
    sys.modules["videofinish"] = package


def load(name):
    return importlib.import_module(f"videofinish.{name}")


def gradient(frames=24, height=64, width=96):
    return torch.linspace(0, 1, frames * height * width * 3).view(frames, height, width, 3)


def tone(seconds=1.0, rate=44100, channels=2):
    t = torch.arange(round(seconds * rate)) / rate
    return {"waveform": torch.stack([.3 * torch.sin(2 * math.pi * (440 + 110 * c) * t) for c in range(channels)])[None],
            "sample_rate": rate}


def decoded_audio(path):
    """Planar float samples [channels, samples] at the stream's own rate."""
    with av.open(str(path)) as source:
        stream = source.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout=stream.layout.name, rate=stream.sample_rate)
        frames = [f for frame in source.decode(audio=0) for f in resampler.resample(frame)]
        frames += resampler.resample(None)
        return np.concatenate([f.to_ndarray() for f in frames], axis=1)


def no_subprocess(test):
    """A regression to subprocess encoding fails, even when FFmpeg happens to be installed."""
    guard = patch.object(subprocess, "Popen", side_effect=AssertionError("external process"))
    guard.start()
    test.addCleanup(guard.stop)


def comfyui_cpu():
    """Import ComfyUI in CPU mode, or raise SkipTest when it is not on PYTHONPATH."""
    import unittest
    try:
        import comfy.options
    except ImportError:
        raise unittest.SkipTest("ComfyUI is not on PYTHONPATH")
    comfy.options.enable_args_parsing()
    argv, sys.argv = sys.argv, [sys.argv[0], "--cpu"]
    try:
        import comfy.cli_args  # noqa: F401
    finally:
        sys.argv = argv
