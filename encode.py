"""Video encoding in-process with PyAV: no ffmpeg executable, pipes or scratch files.

Frames are written one at a time, so memory does not grow with the video's length. A background thread
converts and encodes them while the caller prepares the next frame. The file is written under a hidden
sibling name and renamed into place once complete, so a failed or cancelled write never leaves a
truncated video. Moved here from ComfyUI-FixedSoundtrack's save_video.py.
"""

from dataclasses import dataclass
from fractions import Fraction
import json
import math
import os
import queue
import threading

import av
import numpy as np
import torch
from av.video.reformatter import ColorPrimaries, ColorRange, ColorTrc

CONTAINERS = {"mp4": "mp4", "mkv": "matroska", "webm": "webm"}
VIDEO_ENCODERS = {"h264": "libx264", "h265": "libx265", "av1": "libsvtav1"}
AUDIO_ENCODERS = {"flac": "flac", "opus": "libopus", "aac": "aac"}
COPY = "copy"
# WebM only stores these; MP4 and MKV store every codec offered here.
WEBM_CODECS = {"video": {"av1", "vp9", "vp8"}, "audio": {"opus", "vorbis"}}
# -1 picks these: core Save Video's H.264 and AV1 defaults; 28 is x265's equivalent of x264's 23.
DEFAULT_CRF = {"h264": 23, "h265": 28, "av1": 30}
MAX_CRF = {"h264": 51, "h265": 51, "av1": 63}
PRESETS = ("fast", "medium", "slow", "slower")
SVT_PRESETS = {"fast": 10, "medium": 8, "slow": 6, "slower": 4}  # SVT-AV1 numbers presets 0 (slowest) to 13
MAX_CHANNELS = 8
COLOR_TRANSFERS = {"sRGB": ColorTrc.IEC61966_2_1, "HDR": ColorTrc.ARIB_STD_B67, "HDR PQ": ColorTrc.SMPTE2084}
BT709_NCL, BT2020_NCL = 1, 9
AUDIO_CHUNK = 4096
QUEUED_FRAMES = 4
H265_MIN = 64


@dataclass(frozen=True)
class Settings:
    container: str = "mp4"
    video_codec: str = "h265"
    crf: int = 18
    video_bitrate: int = 0      # kbit/s; > 0 replaces CRF
    bit_depth: int = 8
    color_space: str = "sRGB"
    audio_codec: str = "flac"
    audio_bitrate: int = 192    # kbit/s for AAC and Opus
    preset: str = "medium"


# Opus only maps these layouts; 3 and 4 channels differ from FFmpeg's defaults (2.1 and 4.0).
OPUS_LAYOUTS = {3: "3.0", 4: "quad"}


def layout(channels, audio_codec=None):
    """The layout for 1 to 8 channels: FFmpeg's default (mono, stereo, 2.1, 4.0, 5.0, 5.1, 6.1, 7.1), or the one
    Opus can map. ComfyUI AUDIO carries no layout, so channel order follows these conventions."""
    if not 1 <= channels <= MAX_CHANNELS:
        raise ValueError(f"Audio needs 1 to {MAX_CHANNELS} channels, got {channels}")
    if audio_codec == "opus" and channels in OPUS_LAYOUTS:
        return OPUS_LAYOUTS[channels]
    return av.AudioLayout(f"{channels}c").name


def check_encoders(video_codec, audio_codec=None):
    """Check the libraries PyAV actually uses, not an unrelated ffmpeg executable."""
    missing = []
    for name in (VIDEO_ENCODERS.get(video_codec), AUDIO_ENCODERS.get(audio_codec)):
        if name is None:
            continue
        try:
            av.Codec(name, "w")
        except (ValueError, av.error.FFmpegError):
            missing.append(name)
    if missing:
        raise RuntimeError(f"This PyAV build lacks the {', '.join(missing)} encoder. Choose another codec, "
                           "or install a PyAV build that has it; an ffmpeg executable does not change PyAV's libraries.")


def color_properties(color_space):
    """Core Save Video's tags: BT.709 with the sRGB curve, or BT.2020 with HLG or PQ."""
    hdr = color_space != "sRGB"
    return {"color_primaries": ColorPrimaries.BT2020 if hdr else ColorPrimaries.BT709,
            "color_trc": COLOR_TRANSFERS[color_space],
            "colorspace": BT2020_NCL if hdr else BT709_NCL,
            "color_range": ColorRange.MPEG}


def tag_values(metadata):
    """JSON-encode non-string values, as core Save Video does for prompt and workflow."""
    return {key: value if isinstance(value, str) else json.dumps(value) for key, value in (metadata or {}).items()}


def validate(settings, width, height, audio_codec_in=None):
    """Check settings before anything is written. audio_codec_in: the codec name of a copied audio stream."""
    s = settings
    if s.container not in CONTAINERS:
        raise ValueError(f"Unknown container {s.container!r}; choose {', '.join(CONTAINERS)}")
    if s.video_codec not in VIDEO_ENCODERS:
        raise ValueError(f"Unknown video codec {s.video_codec!r}; choose {', '.join(VIDEO_ENCODERS)}")
    if s.audio_codec not in (*AUDIO_ENCODERS, COPY):
        raise ValueError(f"Unknown audio codec {s.audio_codec!r}; choose {', '.join(AUDIO_ENCODERS)} or copy")
    if s.container == "webm" and s.video_codec not in WEBM_CODECS["video"]:
        raise ValueError("WebM stores AV1 video only here; choose av1, or the mp4 or mkv container")
    if s.container == "webm" and audio_codec_in is not None and audio_codec_in not in WEBM_CODECS["audio"]:
        raise ValueError(f"WebM stores Opus audio only here, not {audio_codec_in}; choose opus, "
                         "or the mp4 or mkv container")
    if not -1 <= s.crf <= MAX_CRF[s.video_codec]:
        raise ValueError(f"{s.video_codec} CRF must be between 0 and {MAX_CRF[s.video_codec]}, or -1 for the default")
    if s.video_bitrate < 0:
        raise ValueError("Video bitrate must be 0 (use CRF) or positive")
    if s.bit_depth not in (8, 10):
        raise ValueError("Bit depth must be 8 or 10")
    if s.color_space not in COLOR_TRANSFERS:
        raise ValueError(f"Unsupported color space {s.color_space!r}")
    if s.preset not in PRESETS:
        raise ValueError(f"Unknown preset {s.preset!r}; choose {', '.join(PRESETS)}")
    if width <= 0 or height <= 0:
        raise ValueError("The video has no frames")
    if width % 2 or height % 2:
        raise ValueError(f"{s.video_codec} 4:2:0 output requires even dimensions, got {width}x{height}")
    if s.video_codec == "h265" and min(width, height) < H265_MIN:
        # PyAV 19's bundled libx265 corrupts memory on some tiny frames (e.g. 48x32) when encoding off the main
        # thread, which is where ComfyUI runs nodes. Refuse them instead of crashing ComfyUI.
        raise ValueError(f"h265 output needs at least {H265_MIN}x{H265_MIN}, got {width}x{height}; choose h264 or av1")


def frame_rate(rate):
    rate = Fraction(rate).limit_denominator(1_000_000)
    if rate <= 0:
        raise ValueError("Frame rate must be positive")
    return rate


def _video_stream(output, width, height, rate, s):
    stream = output.add_stream(VIDEO_ENCODERS[s.video_codec], rate=rate)
    stream.width, stream.height = width, height
    stream.time_base = 1 / rate
    stream.pix_fmt = "yuv420p10le" if s.bit_depth == 10 else "yuv420p"
    for name, value in color_properties(s.color_space).items():
        setattr(stream.codec_context, name, value)
    options = {}
    if s.video_bitrate > 0:
        # Average bitrate: x264/x265 ABR, SVT-AV1 VBR.
        stream.bit_rate = s.video_bitrate * 1000
    else:
        crf = DEFAULT_CRF[s.video_codec] if s.crf < 0 else s.crf
        if s.video_codec == "av1" and crf == 0:
            options["svtav1-params"] = "lossless=1"  # As core: SVT-AV1 has no CRF 0.
        else:
            options["crf"] = str(crf)
    if s.video_codec == "av1":
        options["preset"] = str(SVT_PRESETS[s.preset])
    else:
        options["preset"] = s.preset
    if s.video_codec == "h265":
        options["x265-params"] = "log-level=warning"
        # Apple players and browsers need hvc1, not FFmpeg's default hev1.
        stream.codec_context.codec_tag = "hvc1"
    stream.options = options
    return stream


def _audio_stream(output, s, sample_rate, channels):
    encoder = av.Codec(AUDIO_ENCODERS[s.audio_codec], "w")
    rate = 48000 if s.audio_codec == "opus" else sample_rate
    rates = encoder.audio_rates
    if rates and rate not in rates:
        rate = min(rates, key=lambda value: abs(value - rate))
    stream = output.add_stream(AUDIO_ENCODERS[s.audio_codec], rate=rate, layout=layout(channels, s.audio_codec))
    if s.audio_codec == "flac":
        stream.codec_context.format = "s32"  # FFmpeg's FLAC encoder stores 24-bit samples.
    else:
        stream.bit_rate = s.audio_bitrate * 1000
    return stream


def pixels(image, bit_depth):
    """[H, W, C] float 0..1 (any device) → RGB numpy array: uint16 for 10-bit, uint8 for 8-bit output.

    16-bit RGB keeps the float precision through the conversion to 10-bit YUV; 8-bit output rounds once, here.
    """
    rgb = image[..., :3].float().clamp(0, 1)
    if bit_depth == 10:
        return (rgb * 65535).round().to(torch.int32).cpu().numpy().astype(np.uint16)
    return (rgb * 255).round().to(torch.uint8).cpu().numpy()


class AudioSource:
    """The first audio stream of a file, for copying its packets unchanged.

    origin: the source time (seconds) the output starts at, i.e. its first video frame. Packets are moved onto
    the output's timeline, keeping their offset from the video; packets that end before the origin are dropped."""

    def __init__(self, file, origin=0):
        if hasattr(file, "seek"):
            file.seek(0)
        self.origin = Fraction(origin)
        self.container = av.open(file)
        if not self.container.streams.audio:
            self.container.close()
            raise ValueError("audio copy needs a source file with an audio stream")
        self.stream = self.container.streams.audio[0]
        self.codec = self.stream.codec_context.codec.canonical_name if self.stream.codec_context else None
        self.packets = self._rebased()

    def _rebased(self):
        for packet in self.container.demux(self.stream):
            if not packet.size:                          # the demuxer's empty flush packet
                continue
            start = packet.pts if packet.pts is not None else packet.dts
            if start is None:
                continue
            shift = round(self.origin / packet.time_base)
            # Drop packets that end before the origin; keep those overlapping it, e.g. Opus priming packets
            # with negative timestamps.
            last = start + packet.duration - 1 if packet.duration else start
            if last < shift:
                continue
            if packet.pts is not None:
                packet.pts -= shift
            if packet.dts is not None:
                packet.dts -= shift
            yield packet

    def close(self):
        self.container.close()


class Writer:
    """Write frames to `path` one at a time, with ComfyUI AUDIO encoded or a file's audio stream copied.

    with Writer(path, width, height, fps, settings, audio=...) as w:
        w.write(image)      # [H, W, 3] float 0..1 tensor
    Leaving the block normally finishes the file; an exception removes it. Audio is trimmed to the video's
    duration, as core Save Video does. Pass `audio` (AUDIO dict) or `audio_copy` (file path or BytesIO), not both.
    """

    def __init__(self, path, width, height, rate, settings, *, audio=None, audio_copy=None, audio_origin=0,
                 metadata=None):
        self.path = os.fspath(path)
        self.width, self.height = width, height
        self.rate = frame_rate(rate)
        self.settings = settings
        self.metadata = metadata
        self.written = 0
        self._audio = None
        self._copy = None
        self._error = None
        self._thread = None
        self._output = None
        self._temporary = None
        if audio is not None and audio_copy is not None:
            raise ValueError("pass audio or audio_copy, not both")
        if audio is not None and settings.audio_codec == COPY:
            raise ValueError("audio copy needs a video file with audio as the source; "
                             "this video's audio is a waveform, so choose flac, opus or aac")
        if audio is not None:
            waveform = audio["waveform"][0]
            layout(waveform.shape[0])
            if waveform.shape[1]:
                self._audio = (waveform.float().cpu().contiguous().numpy(), int(audio["sample_rate"]))
        if audio_copy is not None:
            if settings.audio_codec != COPY:
                raise ValueError("audio_copy needs audio_codec copy")
            self._copy = AudioSource(audio_copy, audio_origin)
        try:
            validate(settings, width, height, self._copy.codec if self._copy else
                     settings.audio_codec if self._audio else None)
            check_encoders(settings.video_codec, settings.audio_codec if self._audio else None)
        except BaseException:
            self._close_copy()
            raise

    def __enter__(self):
        folder, name = os.path.split(os.path.abspath(self.path))
        # Reserve a random sibling with normal output permissions, respecting the umask.
        # mkstemp's owner-only mode would survive os.replace and break shared outputs.
        self._temporary = os.path.join(folder, f".{name}.{os.urandom(8).hex()}.partial")
        os.close(os.open(self._temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
        try:
            s = self.settings
            options = {"movflags": "use_metadata_tags+faststart"} if s.container == "mp4" else {}
            self._output = av.open(self._temporary, "w", format=CONTAINERS[s.container], options=options)
            self._output.metadata.update(tag_values(self.metadata))
            self._video = _video_stream(self._output, self.width, self.height, self.rate, s)
            self._sound = None
            if self._audio is not None:
                samples, sample_rate = self._audio
                self._sound = _audio_stream(self._output, s, sample_rate, samples.shape[0])
                self._fed = 0
            elif self._copy is not None:
                self._sound = self._output.add_stream_from_template(template=self._copy.stream, opaque=True)
                self._pending = None
            self._queue = queue.Queue(maxsize=QUEUED_FRAMES)
            self._thread = threading.Thread(target=self._encode, name="videofinish-encoder", daemon=True)
            self._thread.start()
        except BaseException:
            self._abort()
            raise
        return self

    def write(self, image):
        """Queue one [H, W, C] float frame. Raises the encoder's error, if it failed."""
        if tuple(image.shape[:2]) != (self.height, self.width):
            raise ValueError(f"frame is {image.shape[1]}x{image.shape[0]}, expected {self.width}x{self.height}")
        data = pixels(image, self.settings.bit_depth)
        while True:
            self._raise()
            try:
                self._queue.put(data, timeout=0.1)
                break
            except queue.Full:
                continue
        self.written += 1

    def _raise(self):
        if self._error is not None:
            raise RuntimeError(f"video encoder failed: {self._error}") from self._error

    def _video_end(self, frames):
        return frames / self.rate                          # seconds

    def _feed_audio(self, until_seconds, final=False):
        if self._sound is None:
            return
        if self._audio is not None:
            samples, sample_rate = self._audio
            until = min(math.ceil(sample_rate * until_seconds), samples.shape[1])
            name = layout(samples.shape[0], self.settings.audio_codec)
            while self._fed < until:
                end = min(self._fed + AUDIO_CHUNK, until)
                # Packed samples: PyAV cannot build planar frames with 8 channels.
                packed = np.ascontiguousarray(samples[:, self._fed:end].T).reshape(1, -1)
                chunk = av.AudioFrame.from_ndarray(packed, format="flt", layout=name)
                chunk.sample_rate, chunk.pts, chunk.time_base = sample_rate, self._fed, Fraction(1, sample_rate)
                self._output.mux(self._sound.encode(chunk))
                self._fed = end
            if final:
                self._output.mux(self._sound.encode(None))
            return
        # Copied packets: those that start before the end of the video.
        while True:
            packet = self._pending if self._pending is not None else next(self._copy.packets, None)
            self._pending = None
            if packet is None:
                return
            start = packet.pts if packet.pts is not None else packet.dts
            if start * packet.time_base >= until_seconds:
                if not final:
                    self._pending = packet
                return
            packet.stream = self._sound
            self._output.mux(packet)

    def _encode(self):
        s = self.settings
        source_format = "rgb48le" if s.bit_depth == 10 else "rgb24"
        colorspace = BT709_NCL if s.color_space == "sRGB" else BT2020_NCL
        props = color_properties(s.color_space)
        n = 0
        try:
            while True:
                data = self._queue.get()
                if data is None:
                    break
                # Interleave audio with video rather than appending it after all frames.
                self._feed_audio(self._video_end(n + 1))
                frame = av.VideoFrame.from_ndarray(data, format=source_format)
                frame = frame.reformat(format=self._video.pix_fmt, dst_colorspace=colorspace,
                                       dst_color_range=ColorRange.MPEG)
                for name, value in props.items():
                    setattr(frame, name, value)
                frame.pts, frame.time_base = n, 1 / self.rate
                self._output.mux(self._video.encode(frame))
                n += 1
            self._output.mux(self._video.encode(None))
            self._feed_audio(self._video_end(n), final=True)
        except BaseException as error:  # reported to the writing thread
            self._error = error
            while True:                  # unblock a writer waiting on a full queue
                try:
                    if self._queue.get_nowait() is None:
                        break
                except queue.Empty:
                    break

    def __exit__(self, kind, error, traceback):
        if kind is not None:
            self._abort()
            return False
        try:
            self._queue.put(None)
            self._thread.join()
            self._raise()
            if not self.written:
                raise ValueError("no frames were written")
            self._output.close()
            self._output = None
            os.replace(self._temporary, self.path)
        except BaseException:
            self._abort()
            raise
        finally:
            self._close_copy()
        return False

    def _close_copy(self):
        if self._copy is not None:
            self._copy.close()
            self._copy = None

    def _abort(self):
        if self._thread is not None and self._thread.is_alive():
            self._error = self._error or InterruptedError("cancelled")
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            # The encoder sees None or drains on error; give it the end marker until it stops.
            while self._thread.is_alive():
                try:
                    self._queue.put(None, timeout=0.1)
                except queue.Full:
                    pass
                self._thread.join(timeout=0.1)
        if self._output is not None:
            try:
                self._output.close()
            except Exception:
                pass
            self._output = None
        if self._temporary is not None:
            try:
                os.remove(self._temporary)
            except FileNotFoundError:
                pass
        self._close_copy()


def encode(path, images, rate, settings=Settings(), *, audio=None, metadata=None, on_frame=None):
    """Encode [frames, H, W, C] float images and optional ComfyUI AUDIO to path.

    on_frame(written, total) runs after each frame and may raise to cancel.
    """
    if images.ndim != 4 or images.shape[0] == 0 or images.shape[-1] < 3:
        raise ValueError("Expected at least one RGB frame shaped [frames, height, width, channels]")
    total = images.shape[0]
    with Writer(path, images.shape[2], images.shape[1], rate, settings, audio=audio, metadata=metadata) as writer:
        for n in range(total):
            writer.write(images[n])
            if on_frame:
                on_frame(n + 1, total)


def remux(source, path, container, metadata=None):
    """Copy every video and audio stream of `source` (path or BytesIO) into `path` without re-encoding.

    The container must be the source's own: a change of container is not a copy here.
    """
    if hasattr(source, "seek"):
        source.seek(0)
    folder, name = os.path.split(os.path.abspath(os.fspath(path)))
    temporary = os.path.join(folder, f".{name}.{os.urandom(8).hex()}.partial")
    with av.open(source) as src:
        formats = src.format.name.split(",")
        if CONTAINERS[container] not in formats and not (container == "mp4" and "mov" in formats):
            raise ValueError(f"copy keeps the source's container ({src.format.name}); choose that container, "
                             "or a video codec to re-encode")
        streams = [s for s in src.streams if s.type in ("video", "audio") and s.codec_context is not None]
        if container == "webm":
            for s in streams:
                if s.codec_context.codec.canonical_name not in WEBM_CODECS[s.type]:
                    raise ValueError(f"WebM cannot store {s.codec_context.codec.canonical_name}")
        os.close(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
        try:
            options = {"movflags": "use_metadata_tags+faststart"} if container == "mp4" else {}
            with av.open(temporary, "w", format=CONTAINERS[container], options=options) as output:
                output.metadata.update(tag_values(metadata))
                mapping = {}
                for s in streams:
                    out = output.add_stream_from_template(template=s, opaque=True)
                    if container == "mp4" and s.codec_context.codec.canonical_name == "hevc":
                        out.codec_tag = "hvc1"
                    mapping[s] = out
                for packet in src.demux(*streams):
                    if not packet.size:                  # the demuxer's empty flush packets
                        continue
                    packet.stream = mapping[packet.stream]
                    output.mux(packet)
            os.replace(temporary, path)
        except BaseException:
            try:
                os.remove(temporary)
            except FileNotFoundError:
                pass
            raise
