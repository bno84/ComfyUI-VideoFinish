# ComfyUI-VideoFinish — design and implementation plan

Subtitle: *Video upscaling, resizing, interpolation, and configurable export.*

Status: agreed layout (2026-11-09, revised the same day), not yet implemented. This repo is local only: no remote, nothing pushed or
published. History: an earlier draft kept processed frames unencoded in memory. One review round (gpt-6.1-sol,
medium) checked that draft; its findings that still apply are folded in below (marked *review*). That draft was
replaced after measuring memory: about 52 GB of float32 frames for one 43 s H3 video at 1080p/48 fps.

## 1. Layout

Three nodes, all on ComfyUI's native `VIDEO`.

| ID | Display name | In → Out |
|---|---|---|
| `VideoFinishSaveVideo` | Save Video (VideoFinish) | VIDEO → `video` (input, unchanged), `saved` (VIDEO of the written file) |
| `VideoFinishPlan` | Video Finish: post-process plan | options → `VIDEOFINISH_PLAN` |
| `VideoFinishProcess` | Video Finish: process and save | VIDEO + plan (+ optional `same_number_as`) → `saved` (VIDEO of the written file), report |

```
VAE decode ─► Create Video ─► Save Video (VideoFinish) ──── encode 1: intermediate copy
              (tensor VIDEO)     │ video (passthrough)   │ saved (file)
                                 ▼                       ▼
            post-process plan ─► Video Finish: process and save ──── encode 2: processed copy
                                 (frames read as given)  (same_number_as)
```

Each output costs exactly one encode, and both are made from the original frames and soundtrack. The saver's
`video` passthrough is the input itself (no copy, no encode). Connecting post-processing to it makes post run
after the intermediate save, which is ComfyUI's usual way to order nodes that don't depend on each other.

The process node also accepts a file-backed VIDEO (a saved file, the AudioVisualizer output, Load Video). It then
starts from a lossy generation; templates that use that route note it and suggest raising the intermediate's quality.
This was the case behind the hidden extra encode in "Owned by the Cat - Post": FixedSoundtrackSaveVideo passed a
tensor VIDEO on, and VideoPost's ffmpeg pipeline needed a file, so it encoded a temporary copy first.

### Save Video (VideoFinish)

Moved from FixedSoundtrack (`save_video.py` and its tests), and replaces `FixedSoundtrackSaveVideo`.
- Containers mp4/mkv/webm; video h264/h265/av1/**copy**; audio flac/opus/aac/**copy**; CRF or bitrate;
  bit depth auto/8/10; preset; HDR tags from the source VIDEO. In-process PyAV, no executable or subprocess;
  atomic hidden partial file then rename; progress and cancellation; workflow metadata.
- `copy` remuxes a file-backed VIDEO without re-encoding (core `VideoFromFile.save_to` already reuses streams when
  container, codec, depth, CRF, color, trim and crop allow it; verified on the pinned ComfyUI) and errors clearly for a
  tensor VIDEO. For sources that are already final files.
- Naming: `<prefix>_<number>_<suffix>.<ext>`, never overwrites (`-2`, `-3`), `save` toggle, as VideoPostSave.
  Optional player tags (title, artist, comment, lyrics) as in AudioVisualizer.
- Outputs: `video`, the input unchanged (for ordering and lossless frames downstream), and `saved`,
  `VideoFromFile(saved path)`, for chaining file-based nodes and for `same_number_as`. With `save` off, `saved`
  still reserves and carries the number, without writing a file.
- Node defaults suit working copies: H.265 CRF 18, preset medium, bit depth auto, FLAC. Templates set their own
  values per use case, e.g. Opus 128 for distributable copies (FixedSoundtrack template).

### Video Finish: post-process plan

One node replaces the upscale/resize/RIFE stage nodes. Options only, no video input, nothing rendered:
- `upscale` on/off, `upscale_model` from ComfyUI's `upscale_models` (spandrel; the scale is read from the model
  and validated). Tested: Real-CUGAN conservative 4×, LiveAction V1 SPAN 2×.
- `resize` fit width/height/off, `size`. Now also valid **without** upscale (the old "only after upscale" rule is
  dropped). Lanczos-3 with PIL's coefficients on the GPU, float, no 8-bit rounding between passes.
- `interpolate` none/2×/3×/4×, `rife_model` from a `rife` model folder (RIFE 4.26 tested).
- Output: a small `VIDEOFINISH_PLAN` dict (pure Python, as `plan.py` today). Final sizes need the source size,
  so they are resolved by the process node.

### Video Finish: process and save

- Inputs: `video` (any VIDEO; typically the saver's `video` passthrough), `plan`, `suffix` (default `post`), the
  same encode controls as the saver (same code), `filename_prefix`, optional `same_number_as` (VIDEO).
- Numbering: with `same_number_as` connected to a saver's `saved` output, the result gets that file's prefix and
  number with its own suffix (`Cat_00012_.mp4` → `Cat_00012_post.mp4`), and `filename_prefix` is ignored.
  Otherwise the normal counter. This replaces VideoPostSave's `same_as`.
- Streams decode → steps → encode one frame at a time (two for RIFE), as VideoPost does now. CPU memory
  is constant; GPU use is unchanged from today: models unloaded first, upscale tiles fall back from 512 to 256 to 128 px on OOM.
- Source, never encoded first:
  - Tensor VIDEO (Create Video, or the saver's passthrough): the frames from `get_components()`, already in
    memory upstream (at H3's 486×270, 43 s ≈ 1.6 GB float32), fed one at a time.
  - File-backed VIDEO without trim/crop: decoded incrementally by PyAV from `get_stream_source()`.
  - Trimmed/cropped file VIDEO: `get_components()` (materializes the source; documented).
- Audio: a tensor VIDEO's waveform is encoded with the chosen audio codec (FLAC by default). A file source can
  also use `copy`, which adds no audio generation.
- Nothing enabled in the plan: writes nothing and blocks its VIDEO output (`ExecutionBlocker`), with a report.
- Outputs: `saved`, `VideoFromFile(saved path)`, and a report (steps, sizes, FPS, frames, time, frame source).
- RIFE: last frame held (duration unchanged), cuts interpolated (documented). Progress per output frame,
  cancellation per frame, partial file removed on failure.

## 2. Decode, color and timing policy (*review*; applies to file sources)

- Decode with the source's matrix (BT.601/709/2020) and range (limited/full) from stream tags; untagged → BT.709
  limited, noted in the report. 8-bit sources as rgb24, >8-bit as rgb48, to float on the GPU.
- Encode tags follow the source, as VideoPost does today.
- HDR (PQ/HLG) sources: learned upscalers refused (SDR-trained); resize and RIFE allowed. Alpha: refused.
- Rotation honored (multiples of 90°), as today.
- Exact rational FPS (no 1/1000 rounding, unlike the FixedSoundtrack encoder and core); tested with 24000/1001 and
  30000/1001. VFR sources are processed at their average rate, as stated in the report.
- Audio channels 1–8 with FFmpeg default layouts (the visualizer accepts 8; the FixedSoundtrack encoder 1/2/6).

## 3. Repository layout

```
__init__.py         registration
nodes.py            the three nodes
encode.py           PyAV encoder (moved from FixedSoundtrack save_video.py), plus copy/remux
naming.py           numbering, suffix, no-overwrite, number reuse from a source path
plan.py             plan options and size resolution (pure Python)
process.py          decode → steps → encode stream
resize.py           GPU Lanczos (PIL coefficients)
rife/ifnet.py       vendored minimal IFNet 4.26 inference (no import-time device), attribution header
models.py           folder_paths registration: append `rife`; use `upscale_models`
example_workflows/  Create Video → save → passthrough → process example (+ thumbnail)
docs/               DESIGN.md, icon.svg, icon.png, node screenshot
tests/              see §6
.github/workflows/ci.yml, pyproject.toml, README.md, RELEASING.md, AGENTS.md, LICENSE, THIRD_PARTY_NOTICES.md
```

`pyproject.toml`: `comfyui-videofinish` 0.1.0, MIT, `dependencies = []` (av, spandrel, torch, numpy come with
ComfyUI 0.38), `requires-comfyui ">=0.38.0"`, PublisherId bno84, icon URL on `main`.

The ComfyUI-Frame-Interpolation folder import is replaced by vendored code. Sources: Fannovel16 `rife_arch.py`
at `26545cc2` (MIT), derived from hzwer/Practical-RIFE (MIT) and HolyWu/vs-rife (MIT); exact revisions and
licence texts kept verbatim in THIRD_PARTY_NOTICES (*review*). Weights are never bundled; README lists sources,
SHA-256 and licences (LiveAction SPAN is CC-BY-NC-SA-4.0; CUGAN and RIFE weight licences to verify).

## 4. Changes in other repos

- **FixedSoundtrack** (bno84, signed): remove `FixedSoundtrackSaveVideo`, `save_video.py` and its tests; template
  uses `VideoFinishSaveVideo` (H.265 CRF 18, Opus 128 preserved) with its `cnr_id`; README/RELEASING note the
  dependency. Breaking; minor version bump with release note.
- **AudioVisualizer**: no change needed. It saves its own file and returns `VideoFromFile`, which the process node
  accepts directly (one lossy generation, as with any file source). Its encoder stays its own. Optional later: share `encode.py`.
- **sd-cpp**:
  - Mount the repo read-only in `compose.yaml` and add `upscale_models` and `rife` to `extra_model_paths.yaml`.
  - Remove `containers/comfyui-nodes/ComfyUI-VideoPost` and rebuild "Owned by the Cat - Post" on the new nodes
    (Create Video → saver → passthrough → process).
  - H3 v1.3 (Koyan's H3 Timeline workflow): `H3TLJoinFiles` encodes at a fixed H.265 CRF 18 + Opus 128 (our
    build patch; upstream H.264 + AAC) with no quality controls. VideoFinish does not integrate with it: no `frames`
    input in the public pack. Whether to rebuild v1.3 on decode → Create Video, keep it on old local nodes, or retire
    it with the `h3tl_join_frames.py` patch is sd-cpp housekeeping, decided separately.
  - Update `RELEASES.md`, AGENTS.md and the moved tests.
  - Rebuilding the image or restarting ComfyUI needs permission first.
- Existing workflows may break (user decision), except YuE2-Kit's visualizer workflows, which are unaffected.

## 5. What this removes from the earlier draft

No in-memory frame output, no custom VIDEO object, no cross-pack encoder lookup, no compatibility shims, no memory
preflight. CPU memory is constant per frame, as VideoPost was.

## 6. Verification

CPU/CI (tiny random-weight SPAN via spandrel and random-weight IFNet, no downloads; subprocess guard):
- Saver: codec matrix, CRF vs bitrate, 8/10-bit and HDR tags, WebM rules, `copy` bit-exact packets and
  tensor-VIDEO error, metadata (+ `--disable-metadata`), player tags, numbering/suffix/no-overwrite, cancel
  leaves no partial file, umask, output is the saved file.
- Process: one encode only (count encoder opens; no `save_to` of the source, for tensor and file inputs), tensor
  input processed from the original floats, passthrough ordering (post runs after the save), frame count N×M, exact FPS,
  duration within one frame, audio copy bit-exact, A/V sync with a click/flash clip, channel counts 1/2/6/8, no
  audio, audio shorter than video, color tags round trip and BT.601/709 and full/limited patches, 10-bit source
  path, float through resize (no 8-bit step), nothing-enabled blocks without writing, number reuse from source,
  trimmed input path, cancel mid-stream removes the partial file, constant RSS across clip lengths.
- Plan: rules from `tools/tests/test_video_post.py`, resize without upscale.
- Models: unsupported spandrel architecture, wrong scale, missing file names the folder.
- Install: clean ComfyUI without siblings or `/shared-models`; all nodes register; example workflow validates.
- Schema snapshots of the three nodes (input order, defaults, output types) to catch accidental breaks after 0.1.0.

GPU, in this repo's image (throwaway container; ask before stopping services):
- "Owned by the Cat" path, CUGAN → 1080 → RIFE 2×: two encodes total, numbering pair, frame count, 48 fps,
  duration, time per source frame vs VideoPost's 171 ms, peak RSS.
- Visualizer output → process.

## 7. Publication readiness (prepared, not released)

README (purpose, nodes, models with links/hashes/licences, the passthrough wiring and why, the quality note
for file sources, related packs without novelty claims), example workflow with `cnr_id`s and thumbnail, icon
(drafted), THIRD_PARTY_NOTICES, RELEASING.md and CI cloned from siblings, `comfy node validate` passing.
Not done without separate authorization: GitHub repo creation, pushes, tags, Registry token or publishing.

## 8. Implementation order

1. `encode.py` + saver moved from FixedSoundtrack with its tests, then `copy` and output-is-file.
2. `naming.py`, plan node, process node (port VideoPost `render.py` to PyAV decode/encode), vendored RIFE, models.
3. Tests and CI; README, examples, notices.
4. FixedSoundtrack removal + template switch.
5. sd-cpp integration and workflow rebuilds.
6. GPU validation; fill measured numbers into the README.
