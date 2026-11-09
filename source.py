"""Frames, timing, color and audio of a ComfyUI VIDEO, without encoding it first.

A tensor VIDEO (Create Video, or a saver's passthrough) yields its frames as they are. A file VIDEO is decoded
one frame at a time by PyAV with the matrix and range its stream declares, so memory does not grow with its
length. A trimmed or cropped file VIDEO goes through core get_components(), which decodes it whole.
"""

from fractions import Fraction
import io

import av
import numpy as np
import torch
from av.video.reformatter import ColorRange, ColorTrc

# AVColorSpace → swscale matrix. Anything else, including untagged video, converts as BT.709.
MATRICES = {1: 1, 5: 5, 6: 6, 9: 9, 10: 9}                 # BT.709, BT.470BG, SMPTE 170M, BT.2020 NCL/CL
MATRIX_NAMES = {1: "BT.709", 5: "BT.601", 6: "BT.601", 9: "BT.2020", 10: "BT.2020"}
TRANSFERS = {ColorTrc.SMPTE2084: "HDR PQ", ColorTrc.ARIB_STD_B67: "HDR"}


def file_path(video):
    """The file of an untrimmed, uncropped file VIDEO (path or BytesIO), or None."""
    get = getattr(video, "get_stream_source", None)
    if get is None or not hasattr(video, "get_active_trim_window"):
        return None
    # Only a file VIDEO: the base class's get_stream_source() would encode a tensor VIDEO.
    try:
        from comfy_api.latest import InputImpl
        if not isinstance(video, InputImpl.VideoFromFile):
            return None
    except ImportError:
        if type(video).__name__ != "VideoFromFile":
            return None
    if any(video.get_active_trim_window()) or getattr(video, "_VideoFromFile__crop", None):
        return None
    source = get()
    return source if isinstance(source, (str, io.BytesIO)) else None


def bit_depth(stream):
    fmt = stream.codec_context.format
    if fmt is None:
        return 8
    return max(component.bits for component in fmt.components)


class Source:
    """width, height, fps (Fraction), count (frames, estimated for files), bit_depth, color_space
    ("sRGB", "HDR", "HDR PQ"), audio (ComfyUI AUDIO or None), file (path/BytesIO for audio copy or None),
    has_audio, note (what the report should say). frames() yields [H, W, 3] float32 CPU tensors."""

    def frames(self):
        raise NotImplementedError


class TensorSource(Source):
    def __init__(self, video, components=None):
        components = components or video.get_components()
        images = components.images
        if images.ndim != 4 or images.shape[0] == 0:
            raise ValueError("the video has no frames")
        if components.alpha is not None:
            raise ValueError("videos with alpha are not supported; drop the alpha channel first")
        self.images = images
        self.count, self.height, self.width = images.shape[0], images.shape[1], images.shape[2]
        self.fps = Fraction(components.frame_rate)
        self.audio = components.audio
        self.has_audio = self.audio is not None and self.audio["waveform"].shape[-1] > 0
        self.file = None
        self.bit_depth = video.get_bit_depth() if hasattr(video, "get_bit_depth") else 8
        color = video.get_color_space() if hasattr(video, "get_color_space") else "sRGB"
        self.color_space = "sRGB" if color in (None, "auto") else color
        self.note = f"unencoded frames ({images.dtype})"

    def frames(self):
        for image in self.images:
            yield image[..., :3]


class FileSource(Source):
    def __init__(self, file):
        self.file = file
        with self._open() as container:
            if not container.streams.video:
                raise ValueError("the video file has no video stream")
            stream = container.streams.video[0]
            fps = stream.average_rate or stream.guessed_rate or stream.base_rate
            if not fps or fps <= 0:
                raise ValueError("the video has no valid frame rate")
            self.fps = Fraction(fps)
            self.width, self.height = stream.codec_context.width, stream.codec_context.height
            self.bit_depth = bit_depth(stream)
            self.color_space = TRANSFERS.get(stream.codec_context.color_trc, "sRGB")
            count = stream.frames
            if not count:
                duration = (stream.duration * stream.time_base if stream.duration
                            else Fraction(container.duration or 0, av.time_base))
                count = round(duration * self.fps)
            self.count = int(count)
            self.has_audio = bool(container.streams.audio)
            matrix = MATRIX_NAMES.get(stream.codec_context.colorspace)
            first = next(container.decode(video=0), None)
            if first is None:
                raise ValueError("the video contains no decodable frames")
            if first.rotation and int(round(first.rotation / 90)) % 2:
                self.width, self.height = self.height, self.width
            elif first.rotation and first.rotation % 90:
                raise ValueError("only multiples of 90 degrees of video rotation are supported")
            variable = stream.average_rate and stream.guessed_rate and stream.average_rate != stream.guessed_rate
            self.note = (f"decoded file ({stream.codec_context.codec.name}, {self.bit_depth}-bit, "
                         f"{matrix or 'untagged → BT.709'})" + (", variable frame rate: average used" if variable else ""))
        self.audio = None

    def _open(self):
        if hasattr(self.file, "seek"):
            self.file.seek(0)
        return av.open(self.file)

    def load_audio(self):
        """Decode the first audio stream into ComfyUI AUDIO (for re-encoding; copy uses the file)."""
        if not self.has_audio:
            return None
        with self._open() as container:
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="fltp", layout=stream.layout.name, rate=stream.sample_rate)
            chunks = [f.to_ndarray() for frame in container.decode(stream) for f in resampler.resample(frame)]
            chunks += [f.to_ndarray() for f in resampler.resample(None)]
        if not chunks:
            return None
        waveform = torch.from_numpy(np.concatenate(chunks, axis=1)).float()
        return {"waveform": waveform[None], "sample_rate": stream.sample_rate}

    def frames(self):
        with self._open() as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            out_format = "rgb48le" if self.bit_depth > 8 else "rgb24"
            scale = 65535.0 if self.bit_depth > 8 else 255.0
            for frame in container.decode(stream):
                full = frame.color_range == ColorRange.JPEG or frame.format.name.startswith("yuvj")
                rgb = frame.reformat(format=out_format, src_colorspace=MATRICES.get(frame.colorspace, 1),
                                     src_color_range=ColorRange.JPEG if full else ColorRange.MPEG,
                                     dst_color_range=ColorRange.JPEG)
                image = torch.from_numpy(rgb.to_ndarray().astype(np.float32) / scale)
                quarter = int(round(frame.rotation / 90)) % 4 if frame.rotation else 0
                if quarter:
                    image = torch.rot90(image, k=quarter, dims=(0, 1))
                yield image


def open_source(video):
    """The cheapest exact reading of a VIDEO: its tensors, its file decoded incrementally, or get_components()."""
    file = file_path(video)
    if file is not None:
        return FileSource(file)
    source = TensorSource(video)
    if hasattr(video, "get_stream_source") and type(video).__name__ == "VideoFromFile":
        source.note = "decoded whole (trimmed or cropped file)"
    return source
