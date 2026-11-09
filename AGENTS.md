# Agent guidance

This is the standalone ComfyUI VideoFinish pack: video upscaling, resizing,
interpolation and configurable export around ComfyUI's native `VIDEO` type.
The plan and its decisions are in `docs/DESIGN.md`; read it before changing
the architecture. Keep registered node IDs stable once released.

Every node passes a native VIDEO. Each output costs exactly one
encode: never encode a temporary copy of the input. Process frames one at a
time, in float, without 8-bit rounding between steps. Encode and decode
in-process with PyAV: no ffmpeg executable or subprocesses. Do not bundle
model weights; discover them through ComfyUI's `folder_paths`.

Use the bno84 Git identity and signed conventional commits. Commit bodies
explain the change, then `### Why` and relevant `### Testing` notes.
Append `Assisted-by: MODEL-ID` when creating or significantly changing commits.
Do not add redundant Co-Authored-By lines for agents.

Publishing and remote pushes require user authorization. Local development
does not require or access Registry tokens.
