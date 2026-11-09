# ComfyUI-VideoFinish

Video upscaling, resizing, interpolation, and configurable export.

A ComfyUI node pack for finishing a video: learned upscaling (Real-CUGAN, SPAN and other
spandrel models), precise Lanczos resizing, RIFE frame interpolation, and a configurable
**Save Video (VideoFinish)** node. Each operation exists elsewhere; this pack combines them
around ComfyUI's native `VIDEO` type, with a single encode at the final save.

Status: in development, not yet released. See [docs/DESIGN.md](docs/DESIGN.md).
