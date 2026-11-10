<!--
Describe the pack as it is now: key features, usage and caveats.
This is not a changelog. Replace outdated text instead of appending to it.
-->

# Video Finish for ComfyUI

Video upscaling, resizing, interpolation, and configurable export.

Three nodes around ComfyUI's native `VIDEO`:

- **Save Video (VideoFinish)**: Save Video with codec, quality and audio
  choices, or a stream copy. It passes its input on unchanged and also returns
  the file it wrote.
- **Video Finish: post-process plan**: upscale model, Lanczos resize and RIFE
  interpolation options.
- **Video Finish: process and save**: runs the plan one frame at a time on the
  GPU and encodes the result once, straight to the output folder.

Each of these operations exists in other packs too. This pack combines them so that
each saved file costs exactly one encode. Processing streams frames instead of
holding the whole result in memory (exceptions under *Memory and speed*).

## Install

- **ComfyUI-Manager:** search for _Video Finish_ (`comfyui-videofinish`).
- **comfy-cli:** `comfy node install comfyui-videofinish`.

Requires ComfyUI **0.38.0+**. No extra Python packages: PyAV, spandrel, torch
and NumPy come with ComfyUI. Encoding and decoding run in-process through PyAV;
no FFmpeg executable is used.

## Example: save, then post-process

Open **Video Finish - save and post-process** under **ComfyUI-VideoFinish** in
the template browser.

```
Create Video / Load Video ─► Save Video (VideoFinish) ── working copy (encode 1)
                               │ video          │ saved
                               ▼                ▼ same_number_as
          post-process plan ─► Video Finish: process and save ── processed copy (encode 2)
```

- Connect post-processing to the saver's **`video`** output. That is the input
  itself, unencoded, so processing starts from the same frames as the working
  copy. The link also makes processing run after the save.
- Connect **`saved`** to **`same_number_as`**. The processed file then gets the
  same number with its own suffix: `clip_00012_.mp4` and `clip_00012_post.mp4`.
- From a video file (Load Video, or a `saved` output), processing starts from
  its decoded frames, one lossy generation earlier. When you save a file that
  will be processed later, use a lower CRF or 10-bit.

## Save Video (VideoFinish)

| Container | Video | Audio |
| --- | --- | --- |
| `mp4`, `mkv` | `h264`, `h265`, `av1`, `copy` | `flac`, `opus`, `aac`, `copy` |
| `webm` | `av1`, `copy` | `opus`, `copy` |

- Quality by `crf` (default 18; -1 picks the codec's default) or
  `video_bitrate_kbps`. `preset` trades speed for size.
- `bit_depth` `auto` keeps the source's depth (10-bit for 10-bit or HDR
  sources); `10` keeps more of the frames' float precision.
- `flac` is lossless 24-bit audio; Opus is resampled to 48 kHz. 1 to 8 audio
  channels.
- `copy` keeps a video file's streams as they are (same container), adding the
  workflow metadata. For frames, choose a codec.
- Names are `<prefix>_<number>_<suffix>.<ext>`; an existing name gets `-2`,
  `-3`, never overwritten. The workflow is embedded, as in core Save Video.
  Optional `title`, `artist`, `comment` and `lyrics` inputs add player tags.
- Frame rates stay exact (e.g. 24000/1001). H.265 needs at least 64×64.

Node defaults suit working copies: H.265 CRF 18 with FLAC. Choose Opus or AAC
for smaller distributable files.

## Post-processing

Steps run in this order; each is optional:

1. **Upscale** with any single-image model spandrel loads from
   `models/upscale_models/`. The scale is read from the model.
2. **Resize** with Lanczos-3 to a target height or width, keeping the aspect
   ratio (rounded to even pixels). It uses Pillow's coefficients, antialiased
   when shrinking, on the GPU in float, without 8-bit rounding between passes.
   Works with or without an upscale.
3. **Interpolate** with RIFE 4.26: 2×, 3× or 4× the frame rate at the same
   duration. The last frame is held for its interval. Cuts are interpolated
   like any other frame pair.

With nothing enabled, the process node writes nothing and blocks its `saved`
output. Frames stay float32 from source to encoder. Audio is encoded from the
source's waveform, or copied from a source file with `audio_codec` `copy`. If
VRAM runs out, upscaling falls back to 512, 256, then 128 px tiles. HDR (PQ/HLG)
sources cannot be upscaled, because the models are trained on SDR video.

### Memory and speed

The process node holds a few frames at a time, not the whole result. A tensor
input (Create Video) is already in memory upstream; a file input is decoded one
frame at a time. Two exceptions decode a whole input into memory:

- A trimmed or cropped file VIDEO (e.g. from core Video Slice) is decoded by
  ComfyUI into one float32 tensor of frames × width × height × 12 bytes. The
  console states the estimated size first.
- Re-encoding a file's audio decodes the whole soundtrack: about 90 MB for 4
  minutes of 48 kHz stereo. `audio_codec` `copy` does not.

Audio keeps its offset from the first video frame: audio that starts later
stays delayed, and audio before the first frame is dropped. Gaps inside an
audio stream are not filled when re-encoding.

Measured on an RX 9070 XT (16 GB, ROCm) with a 29.25 s, 704×544, 24 fps clip,
processed by Real-CUGAN 4× → height 1080 → RIFE 2× (1404 frames of 1398×1080 at 48 fps, H.265 CRF 18 medium):

| | |
| --- | --- |
| Time | 99 s: 141 ms per source frame, encoding included |
| Peak VRAM (PyTorch) | 3.0 GiB |
| Peak RAM, whole ComfyUI process, file input | 2.9 GiB (1.1 GiB before processing) |
| The same output held as float32 frames | 23.7 GiB |

A tensor input adds its own size, which upstream nodes already hold: here
702 × 704 × 544 × 3 float32 values = 3.0 GiB.

### Models

Weights are not included or downloaded. Tested with:

| File | Folder | Source | Licence |
| --- | --- | --- | --- |
| `RealCUGAN-4x-conservative.pth` (Real-CUGAN 4×) | `models/upscale_models/` | `updated_weights.zip` of the [Real-CUGAN release](https://github.com/bilibili/ailab/releases/tag/Real-CUGAN), `up4x-latest-conservative.pth` renamed | MIT (bilibili) |
| `2xLiveActionV1_SPAN_490000.pth` (SPAN 2×) | `models/upscale_models/` | [OpenModelDB](https://openmodeldb.info/models/2x-LiveActionV1-SPAN), jcj83429 | CC-BY-NC-SA-4.0: non-commercial |
| `rife4.26.pkl` (RIFE 4.26) | `models/rife/` | [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) v4.26 download, `train_log/flownet.pkl` renamed | MIT (hzwer) |

SHA-256 of the tested files:

```
a8c8185def699b0883662a02df0ef2e6db3b0275170b6cc0d28089b64b273427  RealCUGAN-4x-conservative.pth
8b166c75831ea7f694d9058ee9c8df8148af8cc1d2b57e69e6581b15cab572f7  2xLiveActionV1_SPAN_490000.pth
45c7f74156704769dc9f85cfcaf8552e1e926f9399dcfa3a553dee88fac6f53f  rife4.26.pkl
```

Other folders can be added in `extra_model_paths.yaml` under `upscale_models`
and `rife`. Check each model's licence for your use.

## Related packs

[ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation),
core **Upscale Image (using Model)** and
[Lokken VideoTools](https://github.com/LordLocksley/ComfyUI-Lokken-VideoTools)
cover these operations in other ways. This pack's RIFE network is adapted from
ComfyUI-Frame-Interpolation; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Development

```bash
cd tests
PYTHONPATH=/path/to/ComfyUI python -m unittest discover -s . -v
```

The tests use a tiny random SPAN model and random RIFE weights. Set
`VIDEOFINISH_RIFE_CHECKPOINT` and `VIDEOFINISH_RIFE_REFERENCE` to also check the
real checkpoint and the reference architecture (`tests/test_rife.py`). After a
deliberate change to node inputs, regenerate `tests/schema.json` with
`VIDEOFINISH_UPDATE_SCHEMA=1`. Design notes: [docs/DESIGN.md](docs/DESIGN.md).
