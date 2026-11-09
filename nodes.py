import os
import time

from . import encode as E
from . import models
from . import naming
from . import plan as P
from .source import file_path, open_source

try:
    from comfy_execution.graph_utils import ExecutionBlocker
except ImportError:  # older ComfyUI
    from comfy_execution.graph import ExecutionBlocker

CATEGORY = "video/finish"
PLAN = "VIDEOFINISH_PLAN"
VIDEO_CODECS = list(E.VIDEO_ENCODERS)
AUDIO_CODECS = list(E.AUDIO_ENCODERS)


def encode_inputs(copy):
    """The encode controls shared by both saving nodes. copy: offer stream copy of the video."""
    return {
        "container": (list(E.CONTAINERS), {"default": "mp4",
                      "tooltip": "MP4 and MKV store every codec here; WebM stores AV1 with Opus only."}),
        "video_codec": (VIDEO_CODECS + (["copy"] if copy else []), {"default": "h265",
                        "tooltip": "CPU encoders: libx264, libx265 (tagged hvc1) and SVT-AV1."
                                   + (" copy: keep a video file's streams as they are (same container; "
                                      "audio is copied too)." if copy else "")}),
        "crf": ("INT", {"default": 18, "min": -1, "max": 63,
                        "tooltip": "Lower means higher quality and larger files. -1: h264 23, h265 28, av1 30. "
                                   "AV1 0 is lossless. Ignored when video_bitrate_kbps is set."}),
        "video_bitrate_kbps": ("INT", {"default": 0, "min": 0, "max": 500000, "step": 100,
                                       "tooltip": "0 uses CRF. Otherwise an average bitrate in kbit/s, instead of CRF."}),
        "bit_depth": (["auto", "8", "10"], {"default": "auto",
                      "tooltip": "auto keeps the video's depth: 8-bit, or 10-bit for a 10-bit source or HDR. "
                                 "10 keeps more of the float precision."}),
        "preset": (list(E.PRESETS), {"default": "medium",
                   "tooltip": "Encoder speed: slower makes smaller files at the same quality."}),
        "audio_codec": (AUDIO_CODECS + ["copy"], {"default": "flac",
                        "tooltip": "FLAC is lossless (24-bit). Opus is resampled to 48 kHz. "
                                   "copy keeps a video file's audio stream as it is. Ignored without audio."}),
        "audio_bitrate_kbps": ("INT", {"default": 192, "min": 16, "max": 512, "step": 16,
                                       "tooltip": "AAC and Opus bitrate in kbit/s. FLAC ignores it."}),
    }


PLAYER_INFO = {key: ("STRING", {"forceInput": True, "tooltip": tooltip}) for key, tooltip in (
    ("title", "Title tag; players such as Dragon Player show it."),
    ("artist", "Artist tag."),
    ("comment", "Comment tag, such as the music style."),
    ("lyrics", "Lyrics tag."),
)}
HIDDEN = {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}


def metadata(prompt, extra_pnginfo, info):
    from comfy.cli_args import args
    if args.disable_metadata:
        return None
    tags = {**(extra_pnginfo or {}), **({"prompt": prompt} if prompt is not None else {})}
    for key in PLAYER_INFO:
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            tags[key] = value.strip()
    return tags or None


def settings(source, container, video_codec, crf, video_bitrate_kbps, bit_depth, preset, audio_codec,
             audio_bitrate_kbps, color_space=None):
    depth = source.bit_depth if bit_depth == "auto" else int(bit_depth)
    color = color_space or source.color_space
    return E.Settings(container=container, video_codec=video_codec, crf=crf, video_bitrate=video_bitrate_kbps,
                      bit_depth=10 if depth >= 10 or (bit_depth == "auto" and color != "sRGB") else 8,
                      color_space=color, audio_codec=audio_codec, audio_bitrate=audio_bitrate_kbps, preset=preset)


def audio_for(source, settings):
    """(audio, audio_copy) for a Writer: the source's waveform, or its file's audio stream for copy."""
    if not source.has_audio:
        return None, None
    if settings.audio_codec == "copy":
        if source.file is None:
            raise ValueError("audio copy needs a video file as the source; this video's audio is a waveform, "
                             "so choose flac, opus or aac")
        return None, source.file
    return (source.audio if source.file is None else source.load_audio()), None


def progress(total):
    import comfy.model_management as mm
    import comfy.utils
    bar = comfy.utils.ProgressBar(max(total, 1))

    def on_frame(written):
        mm.throw_exception_if_processing_interrupted()
        bar.update_absolute(min(written, max(total, written)), max(total, written))
    return on_frame


def shown(file, subfolder):
    return {"images": [{"filename": file, "subfolder": subfolder, "type": "output"}], "animated": (True,)}


class VideoFinishSaveVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "The video to save, e.g. from Create Video."}),
                "filename_prefix": ("STRING", {"default": "video/ComfyUI",
                                               "tooltip": "May include formatting such as %date:yyyy-MM-dd%."}),
                "suffix": ("STRING", {"default": "", "tooltip": "After the number: <prefix>_00012_<suffix>.mp4"}),
                **encode_inputs(copy=True),
            },
            "optional": dict(PLAYER_INFO),
            "hidden": HIDDEN,
        }

    RETURN_TYPES = ("VIDEO", "VIDEO")
    RETURN_NAMES = ("video", "saved")
    OUTPUT_TOOLTIPS = ("The input video, unchanged and unencoded. Connect post-processing here: it then runs "
                       "after this save, from the original frames.",
                       "The file this node wrote, e.g. for 'same_number_as' or nodes that need a file.")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = ("Save Video with H.264, H.265 or AV1 and FLAC, Opus or AAC audio, by CRF or bitrate, or copy a "
                   "video file's streams. Encodes in-process with PyAV, embedding the workflow. Never overwrites.")

    def save(self, video, filename_prefix, suffix, container, video_codec, crf, video_bitrate_kbps, bit_depth,
             preset, audio_codec, audio_bitrate_kbps, prompt=None, extra_pnginfo=None, **info):
        from comfy_api.latest import InputImpl
        tags = metadata(prompt, extra_pnginfo, info)
        if video_codec == "copy":
            source = file_path(video)
            if source is None:
                raise ValueError("video_codec copy needs a video file (e.g. Load Video) without trim or crop; "
                                 "this video is frames, so choose h264, h265 or av1")
            width, height = video.get_dimensions()
            with naming.reserved(filename_prefix, suffix, container, width, height) as (path, file, subfolder):
                E.remux(source, path, container, metadata=tags)
        else:
            source = open_source(video)
            s = settings(source, container, video_codec, crf, video_bitrate_kbps, bit_depth, preset, audio_codec,
                         audio_bitrate_kbps)
            audio, audio_copy = audio_for(source, s)
            on_frame = progress(source.count)
            with naming.reserved(filename_prefix, suffix, container, source.width, source.height) as (path, file, subfolder):
                with E.Writer(path, source.width, source.height, source.fps, s, audio=audio, audio_copy=audio_copy,
                              audio_origin=source.origin, metadata=tags) as writer:
                    for image in source.frames():
                        writer.write(image)
                        on_frame(writer.written)
        return {"ui": shown(file, subfolder), "result": (video, InputImpl.VideoFromFile(path))}


class VideoFinishPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "upscale": ("BOOLEAN", {"default": False, "tooltip": "on = upscale every frame with the model below"}),
            "upscale_model": ([P.NONE, *models.upscale_models()], {"tooltip": "From models/upscale_models (spandrel). Tested: "
                              "Real-CUGAN conservative 4x, LiveAction V1 SPAN 2x."}),
            "resize": (list(P.FITS), {"default": "off", "tooltip": "Lanczos resize after any upscale, locked aspect "
                       "ratio: 'size' sets this side, the other follows (rounded to even pixels)."}),
            "size": ("INT", {"default": 1080, "min": 64, "max": 8192, "step": 2,
                             "tooltip": "Target pixels for the 'resize' side, e.g. height 1080 or width 1920."}),
            "interpolate": (list(P.INTERPOLATIONS), {"default": "none", "tooltip": "RIFE frame interpolation: "
                            "2x/3x/4x the frame rate, same duration. Runs last."}),
            "rife_model": ([P.NONE, *models.rife_models()], {"tooltip": "From models/rife: RIFE 4.26 (Practical-RIFE flownet.pkl)."}),
        }}

    RETURN_TYPES = (PLAN,)
    RETURN_NAMES = ("plan",)
    FUNCTION = "make"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Post-process options: upscale → Lanczos resize → RIFE interpolation. Nothing is processed "
                   "here; connect the plan to 'Video Finish: process and save'.")

    def make(self, upscale, upscale_model, resize, size, interpolate, rife_model):
        return (P.make(upscale, upscale_model, resize, size, interpolate, rife_model),)


class VideoFinishProcess:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "Best: the 'video' output of Save Video (VideoFinish), so this runs "
                                               "after that save and starts from the original frames. A video file "
                                               "works too, starting from its decoded (lossy) frames."}),
                "plan": (PLAN,),
                "filename_prefix": ("STRING", {"default": "video/ComfyUI",
                                               "tooltip": "Ignored when 'same_number_as' is connected."}),
                "suffix": ("STRING", {"default": "post", "tooltip": "After the number: <prefix>_00012_<suffix>.mp4"}),
                **encode_inputs(copy=False),
            },
            "optional": {
                "same_number_as": ("VIDEO", {"tooltip": "A save's 'saved' output: this file gets the same prefix and "
                                                        "number, with its own suffix."}),
                **PLAYER_INFO,
            },
            "hidden": HIDDEN,
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("saved", "report")
    OUTPUT_TOOLTIPS = ("The file this node wrote. Blocked when the plan enables nothing.",
                       "What was done, with sizes, frame rate, frame count and time.")
    FUNCTION = "process"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = ("Runs the plan's steps one frame at a time on the GPU and encodes the result once, straight to the "
                   "output folder. With nothing enabled it writes nothing and blocks its 'saved' output.")

    def process(self, video, plan, filename_prefix, suffix, container, video_codec, crf, video_bitrate_kbps,
                bit_depth, preset, audio_codec, audio_bitrate_kbps, same_number_as=None, prompt=None,
                extra_pnginfo=None, **info):
        if not P.enabled(plan):
            report = "no post-processing enabled: nothing written"
            print("[video finish] " + report)
            return {"ui": {}, "result": (ExecutionBlocker(None), report)}

        import comfy.model_management as mm
        from comfy_api.latest import InputImpl
        from . import process as R

        source = open_source(video)
        if plan["upscale"] and source.color_space != "sRGB":
            raise ValueError(f"the video is {source.color_space}; upscale models are trained on SDR video. "
                             "Convert it to SDR first, or turn the upscale off.")
        upscaler, rife = R.models_for(plan)
        steps, notes = P.resolve(plan, source.width, source.height, upscaler.scale if upscaler else None)
        out_w, out_h = P.output_size(steps, source.width, source.height)
        rate = R.output_rate(source, steps)
        s = settings(source, container, video_codec, crf, video_bitrate_kbps, bit_depth, preset, audio_codec,
                     audio_bitrate_kbps)
        audio, audio_copy = audio_for(source, s)

        number = naming.numbered(same_number_as) if same_number_as is not None else None
        if same_number_as is not None and number is None:
            notes.append("'same_number_as' is not a numbered output file: counted normally")
        prefix = number[0] if number else filename_prefix
        expected = source.count * P.multiplier(steps)
        on_frame = progress(expected)
        description = P.describe(steps, source.width, source.height, source.fps)
        print(f"[video finish] {description}")
        mm.unload_all_models()                           # give the GPU to the post-process models
        mm.soft_empty_cache()
        started = time.monotonic()
        try:
            with naming.reserved(prefix, suffix, container, out_w, out_h,
                                 number=number[1] if number else None) as (path, file, subfolder):
                with E.Writer(path, out_w, out_h, rate, s, audio=audio, audio_copy=audio_copy,
                              audio_origin=source.origin, metadata=metadata(prompt, extra_pnginfo, info)) as writer:
                    count = R.run(source, steps, upscaler, rife, writer, mm.get_torch_device(), on_frame=on_frame)
        finally:
            del upscaler, rife
            mm.soft_empty_cache()
        elapsed = time.monotonic() - started
        if count != expected:
            notes.append(f"wrote {count} frames, expected {expected} from the source's frame count: "
                         "check audio sync")
        report = "\n".join([
            description,
            f"source: {source.note}",
            f"{count} frames at {float(rate):g} fps in {elapsed / 60:.1f} min → {os.path.join(subfolder, file)}",
            f"{s.video_codec} {'%d kbit/s' % s.video_bitrate if s.video_bitrate else 'CRF %d' % s.crf} "
            f"{s.bit_depth}-bit {s.preset}, audio {s.audio_codec if source.has_audio else 'none'}",
            *(f"- {note}" for note in notes),
        ])
        print("[video finish] " + report.splitlines()[2])
        return {"ui": shown(file, subfolder), "result": (InputImpl.VideoFromFile(path), report)}


NODE_CLASS_MAPPINGS = {
    "VideoFinishSaveVideo": VideoFinishSaveVideo,
    "VideoFinishPlan": VideoFinishPlan,
    "VideoFinishProcess": VideoFinishProcess,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFinishSaveVideo": "Save Video (VideoFinish)",
    "VideoFinishPlan": "Video Finish: post-process plan",
    "VideoFinishProcess": "Video Finish: process and save",
}
