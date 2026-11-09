"""In-process PyAV encoding and decoding: no FFmpeg or FFprobe executable is used."""

from fractions import Fraction
import io
import itertools
import json
import math
import os
import stat
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np
import torch

from helpers import decoded_audio, gradient, load, no_subprocess, tone

ev = load("encode")
src = load("source")
S = ev.Settings


class EncodeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        no_subprocess(self)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def test_every_supported_combination_round_trips(self):
        metadata = {"prompt": {"1": {"class_type": "X"}}, "workflow": {"nodes": []}, "title": "Plain title"}
        for container, video_codec, audio_codec in itertools.product(ev.CONTAINERS, ev.VIDEO_ENCODERS, ev.AUDIO_ENCODERS):
            if container == "webm" and (video_codec, audio_codec) != ("av1", "opus"):
                continue
            with self.subTest(container=container, video=video_codec, audio=audio_codec):
                path = self.path(f"{video_codec}-{audio_codec}.{container}")
                ev.encode(path, gradient(), 24, S(container=container, video_codec=video_codec, crf=-1,
                                                  audio_codec=audio_codec), audio=tone(), metadata=metadata)
                with av.open(path) as source:
                    video, audio = source.streams.video[0], source.streams.audio[0]
                    self.assertEqual(video.codec_context.codec.canonical_name,
                                     {"h264": "h264", "h265": "hevc", "av1": "av1"}[video_codec])
                    self.assertEqual(audio.codec_context.codec.canonical_name, audio_codec)
                    self.assertEqual(audio.sample_rate, 48000 if audio_codec == "opus" else 44100)
                    self.assertEqual(len(audio.layout.channels), 2)
                    if container == "mp4" and video_codec == "h265":
                        self.assertEqual(video.codec_tag, "hvc1")
                    self.assertEqual((video.color_primaries, video.color_trc, video.colorspace), (1, 13, 1))
                    tags = {key.lower(): value for key, value in source.metadata.items()}
                    self.assertEqual(json.loads(tags["prompt"]), metadata["prompt"])
                    self.assertEqual(json.loads(tags["workflow"]), metadata["workflow"])
                    self.assertEqual(tags["title"], "Plain title")
                    self.assertEqual(sum(1 for _ in source.decode(video=0)), 24)
        self.assertFalse([name for name in os.listdir(self.tmp.name) if name.startswith(".")])

    def test_flac_is_lossless_and_trimmed_to_the_video(self):
        audio = tone(seconds=2.0, channels=1)
        path = self.path("lossless.mkv")
        ev.encode(path, gradient(), 24, S(container="mkv", video_codec="h264"), audio=audio)
        samples = decoded_audio(path)
        source = audio["waveform"][0, :, :44100].numpy()
        self.assertEqual(samples.shape, source.shape)
        np.testing.assert_allclose(samples, source, atol=2 ** -22)

    def test_one_to_eight_channels(self):
        for channels, codec in itertools.product(range(1, 9), ("flac", "opus", "aac")):
            with self.subTest(channels=channels, codec=codec):
                path = self.path(f"{channels}-{codec}.mkv")
                ev.encode(path, gradient(frames=6), 24, S(container="mkv", video_codec="h264", audio_codec=codec),
                          audio=tone(seconds=0.25, rate=48000, channels=channels))
                with av.open(path) as source:
                    self.assertEqual(len(source.streams.audio[0].layout.channels), channels)
        with self.assertRaisesRegex(ValueError, "1 to 8 channels"):
            ev.encode(self.path("x.mkv"), gradient(), 24, S(container="mkv"), audio=tone(channels=9))

    def test_exact_rational_frame_rates(self):
        for rate in (Fraction(24000, 1001), Fraction(30000, 1001), Fraction(24), Fraction(60)):
            with self.subTest(rate=rate):
                path = self.path(f"{rate.numerator}.mp4")
                ev.encode(path, gradient(frames=30), rate, S(video_codec="h264"), audio=tone(seconds=2))
                with av.open(path) as source:
                    video = source.streams.video[0]
                    self.assertEqual(video.average_rate, rate)
                    last = list(source.decode(video=0))[-1]
                    self.assertEqual(last.pts * last.time_base, 29 / rate)
                # Audio is cut at the video's exact duration.
                self.assertEqual(decoded_audio(path).shape[1], math.ceil(44100 * 30 / rate))

    def test_audio_shorter_than_video_and_no_audio(self):
        path = self.path("short.mp4")
        ev.encode(path, gradient(frames=48), 24, S(video_codec="h264"), audio=tone(seconds=0.5))
        self.assertEqual(decoded_audio(path).shape[1], 22050)
        path = self.path("silent.mp4")
        ev.encode(path, gradient(), 24, S(video_codec="h264"), audio={"waveform": torch.zeros(1, 2, 0), "sample_rate": 44100})
        with av.open(path) as source:
            self.assertFalse(source.streams.audio)

    def test_ten_bit_and_hdr_tags(self):
        path = self.path("hdr.mkv")
        ev.encode(path, gradient(), 24, S(container="mkv", bit_depth=10, color_space="HDR PQ"))
        with av.open(path) as source:
            video = source.streams.video[0]
            self.assertEqual(video.codec_context.pix_fmt, "yuv420p10le")
            self.assertEqual((video.color_primaries, video.color_trc, video.colorspace), (9, 16, 9))
            self.assertFalse(source.streams.audio)

    def test_ten_bit_keeps_steps_finer_than_eight_bit(self):
        # A ramp with steps of a quarter 8-bit level: 10-bit output keeps them, 8-bit cannot.
        ramp = (torch.arange(64).float() / (4 * 255) + 0.25).view(1, 1, 64, 1).expand(4, 64, 64, 3).contiguous()
        levels = {}
        for depth in (8, 10):
            path = self.path(f"ramp{depth}.mkv")
            ev.encode(path, ramp, 24, S(container="mkv", crf=0, bit_depth=depth, preset="slow"))
            with av.open(path) as source:
                frame = next(source.decode(video=0)).to_ndarray(format="gray16le" if depth == 10 else "gray")
            levels[depth] = len(np.unique(frame[32]))
        self.assertGreater(levels[10], 2 * levels[8])

    def test_video_bitrate_replaces_crf(self):
        noise = torch.rand(48, 96, 128, 3, generator=torch.Generator().manual_seed(1))
        for kbps in (100, 2000):
            path = self.path(f"{kbps}.mp4")
            ev.encode(path, noise, 24, S(video_codec="h264", video_bitrate=kbps))
            # Within a factor of two of the requested average bitrate over two seconds.
            self.assertLess(abs(math.log2(os.path.getsize(path) * 8 / 2 / (kbps * 1000))), 1)

    def test_presets_reach_the_encoder(self):
        sizes = {}
        for codec in ("h264", "h265", "av1"):
            for preset in ("fast", "slower"):
                path = self.path(f"{codec}-{preset}.mkv")
                ev.encode(path, gradient(frames=4), 24, S(container="mkv", video_codec=codec, preset=preset))
                sizes[codec, preset] = os.path.getsize(path)
        self.assertTrue(all(size > 0 for size in sizes.values()))

    def test_invalid_choices_fail_before_writing(self):
        cases = [
            (S(container="webm", video_codec="h265", audio_codec="opus"), "AV1 video only"),
            (S(container="webm", video_codec="av1", audio_codec="flac"), "Opus audio only"),
            (S(video_codec="h264", crf=52), "between 0 and 51"),
            (S(bit_depth=12), "8 or 10"),
            (S(preset="placebo"), "Unknown preset"),
        ]
        for settings, message in cases:
            with self.subTest(settings=settings), self.assertRaisesRegex(ValueError, message):
                ev.encode(self.path("x.mp4"), gradient(), 24, settings, audio=tone())
        with self.assertRaisesRegex(ValueError, "even dimensions"):
            ev.encode(self.path("x.mp4"), gradient(width=95), 24)
        with self.assertRaisesRegex(ValueError, "at least 64x64"):
            ev.encode(self.path("x.mp4"), gradient(height=32, width=48), 24)
        with self.assertRaisesRegex(ValueError, "waveform"):
            ev.encode(self.path("x.mp4"), gradient(), 24, S(audio_codec="copy"), audio=tone())
        self.assertEqual(os.listdir(self.tmp.name), [])

    def test_webm_without_audio_ignores_audio_codec(self):
        ev.encode(self.path("silent.webm"), gradient(), 24, S(container="webm", video_codec="av1", audio_codec="flac"))

    def test_cancel_leaves_no_partial_file(self):
        def cancel(written, total):
            if written == 3:
                raise InterruptedError("cancelled")
        with self.assertRaises(InterruptedError):
            ev.encode(self.path("cancelled.mp4"), gradient(), 24, audio=tone(), on_frame=cancel)
        self.assertEqual(os.listdir(self.tmp.name), [])

    def test_output_respects_umask(self):
        previous = os.umask(0o022)
        try:
            path = self.path("shared.mp4")
            ev.encode(path, gradient(), 24)
        finally:
            os.umask(previous)
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o644)

    def test_missing_encoder_names_the_pyav_build(self):
        real = av.Codec

        def codec(name, mode):
            if name == "libx265":
                raise ValueError("unavailable")
            return real(name, mode)
        with patch.object(ev.av, "Codec", side_effect=codec), self.assertRaisesRegex(RuntimeError, "lacks the libx265"):
            ev.encode(self.path("x.mp4"), gradient(), 24)


class CopyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        no_subprocess(self)
        self.source = os.path.join(self.tmp.name, "source.mp4")
        ev.encode(self.source, gradient(), 24, S(video_codec="h265", audio_codec="opus"), audio=tone(seconds=1))

    def packets(self, path, kind):
        with av.open(path) as container:
            return [bytes(p) for p in container.demux(getattr(container.streams, kind)[0]) if p.size]

    def test_remux_copies_packets_and_adds_metadata(self):
        out = os.path.join(self.tmp.name, "copy.mp4")
        ev.remux(self.source, out, "mp4", metadata={"workflow": {"nodes": [1]}})
        self.assertEqual(self.packets(out, "video"), self.packets(self.source, "video"))
        self.assertEqual(self.packets(out, "audio"), self.packets(self.source, "audio"))
        with av.open(out) as container:
            self.assertEqual(json.loads(container.metadata["workflow"]), {"nodes": [1]})
            self.assertEqual(container.streams.video[0].codec_tag, "hvc1")

    def test_remux_h264_matroska_keeps_every_frame(self):
        path = os.path.join(self.tmp.name, "h264.mkv")
        offset_file(path, 0, 0)
        out = os.path.join(self.tmp.name, "copy.mkv")
        ev.remux(path, out, "mkv")
        self.assertEqual(self.packets(out, "video"), self.packets(path, "video"))
        with av.open(out) as f:
            self.assertEqual(sum(1 for _ in f.decode(video=0)), 24)

    def test_remux_refuses_another_container(self):
        with self.assertRaisesRegex(ValueError, "keeps the source's container"):
            ev.remux(self.source, os.path.join(self.tmp.name, "copy.mkv"), "mkv")
        self.assertEqual(sorted(os.listdir(self.tmp.name)), ["source.mp4"])

    def test_audio_copy_with_new_video(self):
        out = os.path.join(self.tmp.name, "audio-copy.mkv")
        with ev.Writer(out, 96, 64, 24, S(container="mkv", video_codec="h264", audio_codec="copy"),
                       audio_copy=self.source) as writer:
            for image in gradient(frames=12):
                writer.write(image)
        copied = self.packets(out, "audio")
        original = self.packets(self.source, "audio")
        self.assertEqual(copied, original[:len(copied)])
        # Only packets that start within the new, shorter video (0.5 s of 1 s of 20 ms Opus packets).
        self.assertTrue(24 <= len(copied) <= 27, len(copied))

    def test_audio_copy_into_webm_needs_opus(self):
        aac = os.path.join(self.tmp.name, "aac.mp4")
        ev.encode(aac, gradient(), 24, S(video_codec="h264", audio_codec="aac"), audio=tone())
        with self.assertRaisesRegex(ValueError, "Opus audio only"):
            ev.Writer(os.path.join(self.tmp.name, "x.webm"), 96, 64, 24,
                      S(container="webm", video_codec="av1", audio_codec="copy"), audio_copy=aac)


def offset_file(path, video_start, audio_start, frames=24, codec="libx264", container="matroska"):
    """A file whose video and audio start at the given times (seconds), as some cameras and editors write."""
    with av.open(path, "w", format=container) as out:
        video = out.add_stream(codec, rate=24)
        video.width, video.height, video.pix_fmt = 96, 64, "yuv420p"
        audio = out.add_stream("aac", rate=48000, layout="stereo")
        base = round(video_start * 24)
        for n in range(frames):
            frame = av.VideoFrame.from_ndarray(np.full((64, 96, 3), 8 * n, np.uint8), format="rgb24")
            frame.pts, frame.time_base = base + n, Fraction(1, 24)
            out.mux(video.encode(frame))
        out.mux(video.encode(None))
        start = round(audio_start * 48000)
        for k in range(frames * 2000 // 1024):
            chunk = av.AudioFrame.from_ndarray(np.full((2, 1024), 0.1, np.float32), format="fltp", layout="stereo")
            chunk.sample_rate, chunk.pts, chunk.time_base = 48000, start + 1024 * k, Fraction(1, 48000)
            out.mux(audio.encode(chunk))
        out.mux(audio.encode(None))


class TimelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def write(self, source, settings, out):
        with ev.Writer(out, source.width, source.height, source.fps, settings,
                       audio_copy=source.file if settings.audio_codec == "copy" else None,
                       audio=None if settings.audio_codec == "copy" else source.load_audio(),
                       audio_origin=source.origin) as writer:
            for image in source.frames():
                writer.write(image)

    def first_audio(self, path):
        with av.open(path) as f:
            packets = [p for p in f.demux(f.streams.audio[0]) if p.size]
            return len(packets), packets[0].pts * packets[0].time_base

    def test_copied_audio_follows_a_late_video_start(self):
        path = self.path("late.mkv")
        offset_file(path, 5, 5)
        source = src.FileSource(path)
        self.assertEqual(source.origin, 5)
        self.write(source, S(container="mkv", video_codec="h264", audio_codec="copy"), self.path("copy.mkv"))
        count, start = self.first_audio(self.path("copy.mkv"))
        self.assertGreater(count, 40)
        self.assertLess(abs(start), Fraction(1, 100))

    def test_delayed_audio_keeps_its_delay(self):
        path = self.path("delay.mkv")
        offset_file(path, 0, 0.5)
        source = src.FileSource(path)
        self.write(source, S(container="mkv", video_codec="h264", audio_codec="copy"), self.path("copy.mkv"))
        _, start = self.first_audio(self.path("copy.mkv"))
        self.assertAlmostEqual(float(start), 0.5, delta=0.03)
        self.write(source, S(container="mkv", video_codec="h264", audio_codec="flac"), self.path("flac.mkv"))
        samples = decoded_audio(self.path("flac.mkv"))
        onset = int(np.argmax(np.abs(samples[0]) > 0.05))
        self.assertAlmostEqual(onset / 48000, 0.5, delta=0.03)


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_file_source_round_trips_colors(self):
        # Flat color patches survive BT.709 limited-range encoding and decoding within a few levels.
        colors = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0], [.5, .25, .75]])
        images = colors.view(6, 1, 1, 3).expand(6, 64, 64, 3).contiguous()
        for depth in (8, 10):
            with self.subTest(depth=depth):
                path = os.path.join(self.tmp.name, f"patches{depth}.mkv")
                ev.encode(path, images, 24, S(container="mkv", video_codec="h265", crf=0, bit_depth=depth))
                source = src.FileSource(path)
                self.assertEqual((source.width, source.height, source.fps, source.count, source.bit_depth),
                                 (64, 64, 24, 6, depth))
                decoded = torch.stack(list(source.frames()))
                self.assertEqual(decoded.dtype, torch.float32)
                error = (decoded[:, 16:48, 16:48] - images[:, 16:48, 16:48]).abs().max().item()
                self.assertLess(error, 3 / 255 if depth == 8 else 1 / 255)

    def test_full_range_and_bt601_sources(self):
        path = os.path.join(self.tmp.name, "pc601.mkv")
        with av.open(path, "w", format="matroska") as out:
            stream = out.add_stream("libx264", rate=24)
            stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv444p"
            stream.options = {"qp": "0"}
            stream.codec_context.colorspace, stream.codec_context.color_range = 6, 2
            rgb = np.zeros((64, 64, 3), np.uint8)
            rgb[...] = (200, 40, 120)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(
                format="yuv444p", dst_colorspace=6, dst_color_range=2)
            frame.colorspace, frame.color_range = 6, 2
            out.mux(stream.encode(frame))
            out.mux(stream.encode(None))
        image = next(src.FileSource(path).frames())
        np.testing.assert_allclose(image[32, 32].numpy() * 255, (200, 40, 120), atol=2)

    def test_audio_is_decoded_for_reencoding(self):
        path = os.path.join(self.tmp.name, "a.mkv")
        audio = tone(seconds=1, channels=1)
        ev.encode(path, gradient(), 24, S(container="mkv", video_codec="h264"), audio=audio)
        loaded = src.FileSource(path).load_audio()
        self.assertEqual(loaded["sample_rate"], 44100)
        np.testing.assert_allclose(loaded["waveform"][0].numpy(), audio["waveform"][0, :, :44100].numpy(), atol=2 ** -22)

    def test_file_alpha_is_refused(self):
        path = os.path.join(self.tmp.name, "alpha.mkv")
        with av.open(path, "w", format="matroska") as out:
            stream = out.add_stream("ffv1", rate=24)
            stream.width, stream.height, stream.pix_fmt = 64, 64, "bgra"
            frame = av.VideoFrame.from_ndarray(np.zeros((64, 64, 4), np.uint8), format="bgra")
            out.mux(stream.encode(frame))
            out.mux(stream.encode(None))
        with self.assertRaisesRegex(ValueError, "alpha"):
            src.FileSource(path)

    def test_bytesio_source(self):
        path = os.path.join(self.tmp.name, "b.mp4")
        ev.encode(path, gradient(), 24, S(video_codec="h264"))
        with open(path, "rb") as f:
            source = src.FileSource(io.BytesIO(f.read()))
        self.assertEqual(sum(1 for _ in source.frames()), 24)


if __name__ == "__main__":
    unittest.main()
