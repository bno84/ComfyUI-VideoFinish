"""Single streaming pass: source frames → upscale → Lanczos resize → RIFE → encoder, one frame at a time.

Frames go through one at a time (two for RIFE), on the GPU in float; memory does not grow with the video's
length. Adapted from the sd-cpp ComfyUI-VideoPost render, with PyAV in place of ffmpeg pipes.
"""
from fractions import Fraction

from . import plan as P


def interpolate(frames, model, multiplier):
    """Yield N*M frames at M*FPS, holding the last frame for its final interval (duration unchanged).
    Cuts are interpolated like any other interval."""
    previous = None
    for current in frames:
        if previous is not None:
            yield previous
            for step in range(1, multiplier):
                yield model(previous, current, timestep=step / multiplier)
        previous = current
    if previous is not None:
        for _ in range(multiplier):
            yield previous


def upscale(frames, model, dtype, step, device, log=print):
    import comfy.model_management as mm
    import comfy.utils
    import torch
    width, height = step["size"]
    expected = (1, 3, height, width)
    tile = None                                      # whole frames until VRAM runs out, then tiles
    for x in frames:
        while True:
            try:
                if tile is None:
                    y = model(x.to(dtype))
                else:                                # as core Upscale Image (using Model): 32 px blended overlap
                    y = comfy.utils.tiled_scale(x, lambda t: model(t.to(dtype)).float(), tile_x=tile, tile_y=tile,
                                                overlap=32, upscale_amount=step["scale"], output_device=device)
                break
            except mm.OOM_EXCEPTION:
                mm.soft_empty_cache()
                tile = 512 if tile is None else tile // 2
                if tile < 128:
                    raise
                log(f"[video finish] {step['model']}: out of VRAM, continuing with {tile} px tiles")
        if tuple(y.shape) != expected or not torch.isfinite(y).all().item():
            raise RuntimeError(f"{step['model']}: invalid output (shape {tuple(y.shape)} or non-finite pixels)")
        yield y.float().clamp(0, 1)


def models_for(plan):
    """Load the plan's models on the CPU: (upscaler or None, rife or None)."""
    from . import models
    upscaler = models.load_upscaler(plan["upscale"]["model"]) if plan["upscale"] else None
    rife = models.load_rife(plan["interpolate"]["model"]) if plan["interpolate"] else None
    return upscaler, rife


def run(source, steps, upscaler, rife, writer, device, on_frame=None, log=print):
    """Process source.frames() through steps into writer. Returns the number of frames written."""
    import torch
    out_w, out_h = P.output_size(steps, source.width, source.height)
    if (writer.width, writer.height) != (out_w, out_h):
        raise ValueError(f"writer is {writer.width}x{writer.height}, the steps make {out_w}x{out_h}")
    loaded = {}
    try:
        if upscaler is not None:
            dtype = torch.float16 if upscaler.supports_half and device.type != "cpu" else torch.float32
            loaded["upscale"] = (upscaler.to(device=device, dtype=dtype), dtype)
        if rife is not None:
            loaded["rife"] = rife.to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            frames = (image.to(device, non_blocking=True).permute(2, 0, 1).unsqueeze(0).float()
                      for image in source.frames())
            for step in steps:
                if step["op"] == "upscale":
                    model, dtype = loaded["upscale"]
                    frames = upscale(frames, model, dtype, step, device, log)
                elif step["op"] == "resize":
                    from .resize import Resizer
                    frames = map(Resizer(*step["size"]), frames)
                elif step["op"] == "interpolate":
                    frames = interpolate(frames, loaded["rife"], step["multiplier"])
            count = 0
            for x in frames:
                if tuple(x.shape) != (1, 3, out_h, out_w):
                    raise RuntimeError(f"frame is {tuple(x.shape)}, expected {out_w}x{out_h}")
                writer.write(x[0].permute(1, 2, 0))
                count += 1
                if on_frame is not None:
                    on_frame(count)
        return count
    finally:
        loaded.clear()


def output_rate(source, steps):
    return Fraction(source.fps) * P.multiplier(steps)
