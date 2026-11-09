"""Output names: <prefix>_<number>_<suffix>.<ext>, never overwriting, with a number shared between files.

ComfyUI's counter reads "00012_post" as 12, so suffixed files keep the sequence going.
"""
import os
import re

import folder_paths

NAMED = re.compile(r"^(?P<name>.+)_(?P<number>\d{5,})_(?P<suffix>[^/]*?)(?:-\d+)?\.[A-Za-z0-9]+$")


def numbered(video):
    """(prefix, number) of a file this pack wrote into the output folder, or None.

    video: a VIDEO whose stream source is such a file (a saver's `saved` output)."""
    from comfy_api.latest import InputImpl
    # Only a file VIDEO: the base class's get_stream_source() would encode a tensor VIDEO to find out.
    if not isinstance(video, InputImpl.VideoFromFile):
        return None
    path = video.get_stream_source()
    if not isinstance(path, str):
        return None
    output = os.path.abspath(folder_paths.get_output_directory())
    path = os.path.abspath(path)
    if os.path.commonpath([output, path]) != output:
        return None
    relative = os.path.relpath(path, output)
    match = NAMED.match(os.path.basename(relative))
    if not match:
        return None
    subfolder = os.path.dirname(relative)
    prefix = os.path.join(subfolder, match["name"]) if subfolder else match["name"]
    return prefix, int(match["number"])


def reserve(prefix, suffix, extension, width, height, number=None):
    """A free output path. Returns (path, filename, subfolder). number: reuse this number instead of counting."""
    folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
        prefix, folder_paths.get_output_directory(), width, height)
    os.makedirs(folder, exist_ok=True)
    stem = f"{name}_{counter if number is None else number:05}_{suffix}"
    file, k = f"{stem}.{extension}", 2
    while os.path.lexists(os.path.join(folder, file)):
        file, k = f"{stem}-{k}.{extension}", k + 1
    return os.path.join(folder, file), file, subfolder
