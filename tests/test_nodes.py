"""The three nodes under ComfyUI on CPU, with a tiny random SPAN upscaler and a random-weight RIFE.

Needs ComfyUI on PYTHONPATH; skipped otherwise. No model downloads and no FFmpeg executable.
"""

from fractions import Fraction
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np
import torch

from helpers import comfyui_cpu, decoded_audio, load, no_subprocess, tone

SCHEMA = Path(__file__).with_name("schema.json")


def setUpModule():
    comfyui_cpu()
    global N, E, folder_paths, InputImpl, Types, VideoInput
    import folder_paths
    from comfy_api.latest import InputImpl, Types
    from comfy_api.latest._input import VideoInput
    N = load("nodes")
    E = load("encode")
    global TMP
    TMP = tempfile.TemporaryDirectory()
    root = Path(TMP.name)
    (root / "upscale_models").mkdir()
    (root / "rife").mkdir()
    from spandrel.architectures.SPAN import SPAN
    torch.manual_seed(0)
    torch.save(SPAN(num_in_ch=3, num_out_ch=3, feature_channels=16, upscale=2).state_dict(),
               root / "upscale_models" / "tiny-span-2x.pth")
    torch.save(load("rife.ifnet").IFNet().state_dict(), root / "rife" / "random-rife.pkl")
    folder_paths.add_model_folder_path("upscale_models", str(root / "upscale_models"))
    load("models").register()
    folder_paths.add_model_folder_path("rife", str(root / "rife"))


def tearDownModule():
    TMP.cleanup()


def ramp(frames=12, height=64, width=96):
    """Smooth float frames with steps finer than 8-bit, moving over time."""
    x = torch.linspace(0, 1, width).view(1, 1, width, 1)
    y = torch.linspace(0, 1, height).view(1, height, 1, 1)
    t = torch.arange(frames).float().view(frames, 1, 1, 1) / (4 * 255)
    return (0.2 + 0.5 * x * y + t + torch.tensor([0.0, 0.1, 0.2])).clamp(0, 1).expand(frames, height, width, 3).contiguous()


def tensor_video(images, audio=None, rate=24):
    return InputImpl.VideoFromComponents(Types.VideoComponents(images=images, audio=audio, frame_rate=Fraction(rate)))


SAVE = dict(filename_prefix="video/test", suffix="", container="mp4", video_codec="h265", crf=18,
            video_bitrate_kbps=0, bit_depth="auto", preset="fast", audio_codec="flac", audio_bitrate_kbps=192)
PROCESS = {**SAVE, "suffix": "post"}


class NodeTestCase(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.TemporaryDirectory()
        self.addCleanup(self.out.cleanup)
        previous = folder_paths.get_output_directory()
        folder_paths.set_output_directory(self.out.name)
        self.addCleanup(folder_paths.set_output_directory, previous)
        no_subprocess(self)
        # No hidden encode of an input video: save_to and the base get_stream_source must not run.
        for name in ("save_to", "get_stream_source"):
            guard = patch.object(InputImpl.VideoFromComponents, name, side_effect=AssertionError(f"input {name}"))
            guard.start()
            self.addCleanup(guard.stop)

    def files(self):
        return sorted(str(p.relative_to(self.out.name)) for p in Path(self.out.name).rglob("*") if p.is_file())

    def save(self, video, **kwargs):
        return N.VideoFinishSaveVideo().save(video, **{**SAVE, **kwargs},
                                             prompt={"1": {"class_type": "VideoFinishSaveVideo"}},
                                             extra_pnginfo={"workflow": {"nodes": []}})

    def process(self, video, plan, **kwargs):
        return N.VideoFinishProcess().process(video, plan, **{**PROCESS, **kwargs})


class SaveTests(NodeTestCase):
    def test_passes_the_input_through_and_returns_the_file(self):
        video = tensor_video(ramp(), tone(seconds=0.5))
        result = self.save(video, title="A title")
        passthrough, saved = result["result"]
        self.assertIs(passthrough, video)
        self.assertIsInstance(saved, InputImpl.VideoFromFile)
        self.assertEqual(self.files(), ["video/test_00001_.mp4"])
        self.assertEqual(result["ui"]["images"][0], {"filename": "test_00001_.mp4", "subfolder": "video", "type": "output"})
        with av.open(saved.get_stream_source()) as f:
            self.assertEqual(f.streams.video[0].codec_context.codec.canonical_name, "hevc")
            self.assertEqual(f.streams.audio[0].codec_context.codec.canonical_name, "flac")
            self.assertEqual(json.loads(f.metadata["workflow"]), {"nodes": []})
            self.assertEqual(f.metadata["title"], "A title")

    def test_never_overwrites_and_suffixes(self):
        video = tensor_video(ramp())
        self.save(video)
        self.save(video, suffix="alt")
        self.save(video)
        self.assertEqual(self.files(), ["video/test_00001_.mp4", "video/test_00002_alt.mp4", "video/test_00003_.mp4"])

    def test_copy_keeps_packets_and_refuses_frames(self):
        saved = self.save(tensor_video(ramp(), tone(seconds=0.5)))["result"][1]
        copied = self.save(saved, video_codec="copy", suffix="copy")["result"][1]

        def packets(video):
            with av.open(video.get_stream_source()) as f:
                return [bytes(p) for p in f.demux() if p.size]
        self.assertEqual(packets(copied), packets(saved))
        with self.assertRaisesRegex(ValueError, "copy needs a video file"):
            self.save(tensor_video(ramp()), video_codec="copy")

    def test_reencodes_a_file_without_materializing_it(self):
        saved = self.save(tensor_video(ramp(), tone(seconds=0.5)))["result"][1]
        with patch.object(InputImpl.VideoFromFile, "get_components", side_effect=AssertionError("materialized")):
            again = self.save(saved, video_codec="h264", audio_codec="copy", suffix="h264")["result"][1]
        with av.open(again.get_stream_source()) as f:
            self.assertEqual(f.streams.video[0].frames or sum(1 for _ in f.decode(video=0)), 12)
            self.assertEqual(f.streams.audio[0].codec_context.codec.canonical_name, "flac")

    def test_metadata_can_be_disabled(self):
        from comfy.cli_args import args
        with patch.object(args, "disable_metadata", True):
            saved = self.save(tensor_video(ramp()))["result"][1]
        with av.open(saved.get_stream_source()) as f:
            self.assertNotIn("workflow", f.metadata)


class ProcessTests(NodeTestCase):
    def plan(self, **kwargs):
        options = dict(upscale=False, upscale_model="tiny-span-2x.pth", resize="off", size=1080,
                       interpolate="none", rife_model="random-rife.pkl")
        return N.VideoFinishPlan().make(**{**options, **kwargs})[0]

    def test_cat_workflow_two_encodes_from_the_original_frames(self):
        video = tensor_video(ramp(), tone(seconds=0.5))
        opened = []
        real = E.Writer.__enter__

        def counting(writer):
            opened.append(writer.path)
            return real(writer)
        with patch.object(E.Writer, "__enter__", counting):
            passthrough, saved = self.save(video, suffix="")["result"]
            result = self.process(passthrough, self.plan(upscale=True, resize="width", size=160, interpolate="2x"),
                                  same_number_as=saved)
        self.assertEqual(len(opened), 2)
        processed, report = result["result"]
        self.assertEqual(self.files(), ["video/test_00001_.mp4", "video/test_00001_post.mp4"])
        with av.open(processed.get_stream_source()) as f:
            stream = f.streams.video[0]
            self.assertEqual((stream.codec_context.width, stream.codec_context.height), (160, 106))
            self.assertEqual(stream.average_rate, 48)
            frames = list(f.decode(video=0))
            self.assertEqual(len(frames), 24)
            self.assertEqual(frames[-1].pts * frames[-1].time_base, Fraction(23, 48))
        self.assertEqual(decoded_audio(processed.get_stream_source()).shape[1], 22050)
        self.assertIn("unencoded frames", report)
        self.assertIn("24 frames at 48 fps", report)

    def test_frames_stay_float_between_steps(self):
        written = []
        real = E.Writer.write

        def capture(writer, image):
            written.append(image.detach().cpu().clone())
            return real(writer, image)
        with patch.object(E.Writer, "write", capture):
            self.process(tensor_video(ramp()), self.plan(resize="width", size=192))
        frames = torch.stack(written)
        self.assertEqual(frames.dtype, torch.float32)
        off_grid = ((frames * 255) - (frames * 255).round()).abs() > 0.05
        self.assertGreater(off_grid.float().mean().item(), 0.5)

    def test_nothing_enabled_writes_nothing(self):
        result = self.process(tensor_video(ramp()), self.plan())
        from comfy_execution.graph_utils import ExecutionBlocker
        self.assertIsInstance(result["result"][0], ExecutionBlocker)
        self.assertEqual(self.files(), [])

    def test_file_source_with_audio_copy(self):
        saved = self.save(tensor_video(ramp(), tone(seconds=0.5)), audio_codec="opus")["result"][1]
        processed = self.process(saved, self.plan(interpolate="3x"), audio_codec="copy")["result"][0]

        def audio_packets(video):
            with av.open(video.get_stream_source()) as f:
                return [bytes(p) for p in f.demux(f.streams.audio[0]) if p.size]
        self.assertEqual(audio_packets(processed), audio_packets(saved))
        with av.open(processed.get_stream_source()) as f:
            self.assertEqual(sum(1 for _ in f.decode(video=0)), 36)

    def test_audio_stays_in_sync(self):
        # A white flash at 0.5 s and a click at 0.5 s; after RIFE 2x both stay at 0.5 s.
        images = torch.zeros(24, 64, 96, 3)
        images[12] = 1
        rate = 48000
        waveform = torch.zeros(1, 2, rate)
        waveform[0, :, rate // 2:rate // 2 + 48] = 0.9
        saved = self.process(tensor_video(images, {"waveform": waveform, "sample_rate": rate}),
                             self.plan(interpolate="2x"))["result"][0]
        with av.open(saved.get_stream_source()) as f:
            flash = next(frame.pts * frame.time_base for frame in f.decode(video=0)
                         if frame.to_ndarray(format="gray").mean() > 200)
        samples = decoded_audio(saved.get_stream_source())
        click = Fraction(int(np.argmax(np.abs(samples[0]) > 0.5)), rate)
        self.assertEqual(flash, Fraction(1, 2))
        self.assertLessEqual(abs(click - flash), Fraction(1, 48))

    def test_cancel_removes_the_partial_file(self):
        import comfy.model_management as mm
        calls = []

        def interrupt():
            calls.append(1)
            if len(calls) == 3:
                raise mm.InterruptProcessingException()
        with patch.object(mm, "throw_exception_if_processing_interrupted", interrupt), \
                self.assertRaises(mm.InterruptProcessingException):
            self.process(tensor_video(ramp(frames=24)), self.plan(interpolate="2x"))
        self.assertEqual(self.files(), [])
        self.assertFalse([p for p in Path(self.out.name).rglob(".*")])

    def test_hdr_sources_refuse_upscaling(self):
        video = InputImpl.VideoFromComponents(Types.VideoComponents(images=ramp(), frame_rate=Fraction(24)),
                                              bit_depth=10, color_space="HDR PQ")
        with self.assertRaisesRegex(ValueError, "SDR"):
            self.process(video, self.plan(upscale=True))
        saved = self.process(video, self.plan(interpolate="2x"))["result"][0]
        with av.open(saved.get_stream_source()) as f:
            self.assertEqual(f.streams.video[0].codec_context.pix_fmt, "yuv420p10le")

    def test_missing_and_wrong_models(self):
        with self.assertRaisesRegex(FileNotFoundError, "not found"):
            self.process(tensor_video(ramp()), self.plan(upscale=True, upscale_model="missing.pth"))
        bad = Path(TMP.name) / "upscale_models" / "not-a-model.pth"
        torch.save({"weight": torch.zeros(3)}, bad)
        with self.assertRaisesRegex(ValueError, "spandrel"):
            self.process(tensor_video(ramp()), self.plan(upscale=True, upscale_model="not-a-model.pth"))
        bad_rife = Path(TMP.name) / "rife" / "not-rife.pkl"
        torch.save({"weight": torch.zeros(3)}, bad_rife)
        with self.assertRaisesRegex(ValueError, "RIFE 4.26"):
            self.process(tensor_video(ramp()), self.plan(interpolate="2x", rife_model="not-rife.pkl"))

    def test_counts_normally_without_a_numbered_file(self):
        result = self.process(tensor_video(ramp()), self.plan(interpolate="2x"), same_number_as=tensor_video(ramp()))
        self.assertIn("counted normally", result["result"][1])
        self.assertEqual(self.files(), ["video/test_00001_post.mp4"])


class SchemaTest(unittest.TestCase):
    """Released node IDs, inputs, defaults and outputs change only deliberately: update tests/schema.json."""

    def current(self):
        def describe(cls):
            inputs = cls.INPUT_TYPES()
            return {
                "inputs": {section: {name: [spec[0] if isinstance(spec[0], str) else "COMBO",
                                            (spec[1] if len(spec) > 1 else {}).get("default")]
                                     for name, spec in inputs.get(section, {}).items()}
                           for section in ("required", "optional")},
                "outputs": list(zip(cls.RETURN_TYPES, getattr(cls, "RETURN_NAMES", cls.RETURN_TYPES))),
            }
        return {node: describe(cls) for node, cls in N.NODE_CLASS_MAPPINGS.items()}

    def test_schema_is_unchanged(self):
        current = json.loads(json.dumps(self.current()))
        if os.environ.get("VIDEOFINISH_UPDATE_SCHEMA"):
            SCHEMA.write_text(json.dumps(current, indent=2) + "\n")
        self.assertEqual(current, json.loads(SCHEMA.read_text()))


if __name__ == "__main__":
    unittest.main()
