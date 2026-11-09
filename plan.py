"""Which post-processing steps run, and at what size. Pure Python: no ComfyUI or torch imports.

A plan holds options only. The process node resolves it against the source video and the upscale model's scale,
in the fixed order upscale → resize → interpolate.
"""

FITS = ("off", "height", "width")
INTERPOLATIONS = ("none", "2x", "3x", "4x")
NONE = "none"                                       # model choice when a step needs none


def even(x):
    """Nearest even pixel count: YUV 4:2:0 needs even dimensions."""
    return max(2, 2 * round(x / 2))


def make(upscale=False, upscale_model=None, fit="off", size=1080, interpolate="none", rife_model=None):
    if fit not in FITS:
        raise ValueError(f"Unknown fit {fit!r}; choose {', '.join(FITS)}")
    if interpolate not in INTERPOLATIONS:
        raise ValueError(f"Unknown interpolation {interpolate!r}; choose {', '.join(INTERPOLATIONS)}")
    if upscale and upscale_model in (None, "", NONE):
        raise ValueError("upscale is on: choose an upscale model (models/upscale_models), or turn the upscale off")
    if interpolate != "none" and rife_model in (None, "", NONE):
        raise ValueError("interpolation is on: choose a RIFE model (models/rife), or set interpolate to none")
    return {
        "upscale": {"model": upscale_model} if upscale else None,
        "resize": {"fit": fit, "size": int(size)} if fit != "off" else None,
        "interpolate": {"multiplier": int(interpolate.rstrip("x")), "model": rife_model} if interpolate != "none" else None,
    }


def enabled(plan):
    return any(plan[key] is not None for key in ("upscale", "resize", "interpolate"))


def resolve(plan, width, height, upscale_scale=None):
    """The steps for a width x height source: [{"op": ..., "size": (w, h), ...}] and notes on skipped ones.
    upscale_scale: the loaded model's native scale (needed when the plan upscales)."""
    steps, notes = [], []
    w, h = width, height
    if plan["upscale"] is not None:
        if not upscale_scale:
            raise ValueError("resolving an upscale needs the model's scale")
        w, h = w * upscale_scale, h * upscale_scale
        steps.append({"op": "upscale", "model": plan["upscale"]["model"], "scale": upscale_scale, "size": (w, h)})
    else:
        notes.append("upscale off")
    resize = plan["resize"]
    if resize is not None:
        target = resize["size"]
        if resize["fit"] == "width":
            new = (even(target), even(h * target / w))
        else:
            new = (even(w * target / h), even(target))
        if new == (w, h):
            notes.append(f"resize skipped: already {w}x{h}")
        else:
            steps.append({"op": "resize", "size": new})
            w, h = new
    else:
        notes.append("resize off")
    if plan["interpolate"] is not None:
        steps.append({"op": "interpolate", "multiplier": plan["interpolate"]["multiplier"],
                      "model": plan["interpolate"]["model"], "size": (w, h)})
    else:
        notes.append("interpolation off")
    if (w % 2 or h % 2) and steps:
        raise ValueError(f"the result would be {w}x{h}; 4:2:0 video needs even sizes, so add a resize")
    return steps, notes


def output_size(steps, width, height):
    return steps[-1]["size"] if steps else (width, height)


def multiplier(steps):
    m = 1
    for step in steps:
        if step["op"] == "interpolate":
            m *= step["multiplier"]
    return m


def describe(steps, width, height, fps):
    from fractions import Fraction
    fps = Fraction(fps)
    parts = [f"{width}x{height} @ {float(fps):g} fps"]
    for step in steps:
        if step["op"] == "upscale":
            parts.append(f"{step['model']} ({step['scale']}x) → {step['size'][0]}x{step['size'][1]}")
        elif step["op"] == "resize":
            parts.append(f"lanczos → {step['size'][0]}x{step['size'][1]}")
        elif step["op"] == "interpolate":
            fps *= step["multiplier"]
            parts.append(f"RIFE {step['multiplier']}x → {float(fps):g} fps")
    return " → ".join(parts)
