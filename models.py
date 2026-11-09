"""Model discovery through ComfyUI's folder_paths: upscale models in `upscale_models`, RIFE in `rife`.

Weights are never bundled or downloaded; see the README for sources. extra_model_paths.yaml may add folders
for either name.
"""
import os

import folder_paths

RIFE = "rife"
RIFE_EXTENSIONS = {".pkl", ".pth", ".pt", ".safetensors"}


def register():
    """Add models/rife to ComfyUI's search paths, keeping any folders already configured for it."""
    folder_paths.add_model_folder_path(RIFE, os.path.join(folder_paths.models_dir, RIFE))
    paths, extensions = folder_paths.folder_names_and_paths[RIFE]
    if not extensions or not isinstance(extensions, set):
        folder_paths.folder_names_and_paths[RIFE] = (paths, set(RIFE_EXTENSIONS))
    else:
        extensions.update(RIFE_EXTENSIONS)


def upscale_models():
    return folder_paths.get_filename_list("upscale_models")


def rife_models():
    return folder_paths.get_filename_list(RIFE)


def _path(kind, name):
    path = folder_paths.get_full_path(kind, name) if name else None
    if path is None:
        folders = ", ".join(folder_paths.get_folder_paths(kind)) or f"models/{kind}"
        raise FileNotFoundError(f"{kind} model {name!r} not found; put it in {folders}")
    return path


def load_upscaler(name):
    """A spandrel image-to-image model, as core Load Upscale Model loads it (safe tensor loading)."""
    import comfy.utils
    from spandrel import ImageModelDescriptor, ModelLoader
    state = comfy.utils.load_torch_file(_path("upscale_models", name), safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in state:
        state = comfy.utils.state_dict_prefix_replace(state, {"module.": ""})
    try:
        model = ModelLoader().load_from_state_dict(state).eval()
    except Exception as error:
        raise ValueError(f"{name}: not an upscale model spandrel recognizes ({error})") from error
    if not isinstance(model, ImageModelDescriptor):
        raise ValueError(f"{name}: not a single-image model (spandrel reports {type(model).__name__})")
    if model.input_channels != 3 or model.output_channels != 3:
        raise ValueError(f"{name}: needs 3 input and output channels, has {model.input_channels}/{model.output_channels}")
    return model


def load_rife(name):
    """RIFE 4.26's IFNet from Practical-RIFE's flownet.pkl (any name in the rife folder)."""
    import torch
    from .rife.ifnet import IFNet
    state = torch.load(_path(RIFE, name), map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    # The upstream checkpoint also carries these training-only heads.
    state = {k: v for k, v in state.items() if not k.startswith(("teacher.", "caltime."))}
    model = IFNet()
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"{name}: not a RIFE 4.26 checkpoint ({str(error).splitlines()[0]})") from error
    return model.eval()
