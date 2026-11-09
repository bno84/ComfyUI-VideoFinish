#!/usr/bin/env python3
"""Start ComfyUI on CPU with this pack installed and check that it loads.

Fails when the pack reports an import failure, when an expected node type is
not registered, when an expected GET route does not answer 200, or when an
example workflow uses a node type from comfy-core, this pack or a declared
dependency that the running ComfyUI lacks. Node types of other packs are
reported and skipped.
A backend node without a pack entry (cnr_id) also fails: ComfyUI only adds
those entries when ComfyUI-Manager is installed, so they can go missing.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PORT = 8188
FRONTEND_ONLY = {"MarkdownNote", "Note", "Reroute", "PrimitiveNode"}


def get(path):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=10) as response:
        return response.status, response.read()


def workflow_types(path):
    """Yield (node type, cnr_id) for every backend node, including inside subgraphs."""
    workflow = json.loads(Path(path).read_text())
    subgraphs = workflow.get("definitions", {}).get("subgraphs", [])
    wrappers = {graph["id"] for graph in subgraphs}
    for graph in [workflow, *subgraphs]:
        for node in graph["nodes"]:
            if node["type"] not in FRONTEND_ONLY and node["type"] not in wrappers:
                yield node["type"], (node.get("properties") or {}).get("cnr_id")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comfyui", type=Path, help="ComfyUI checkout with the pack in custom_nodes/")
    parser.add_argument("--pack", required=True, help="Pack directory name in custom_nodes/")
    parser.add_argument("--cnr-id", required=True, help="This pack's registry ID (pyproject name)")
    parser.add_argument("--node", action="append", default=[], help="Node type that must be registered")
    parser.add_argument("--route", action="append", default=[], help="GET path that must answer 200")
    parser.add_argument("--workflow", action="append", default=[], help="Editor workflow to check")
    parser.add_argument("--dependency", action="append", default=[], metavar="CNR_ID=DIR",
                        help="Another pack in custom_nodes/ to load and check workflow nodes against")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--base-directory", type=Path)
    args = parser.parse_args()
    global PORT
    PORT = args.port
    dependencies = dict(item.split("=", 1) for item in args.dependency)
    packs = [args.pack, *dependencies.values()]

    log_path = Path("comfyui.log")
    with log_path.open("w") as log:
        server = subprocess.Popen(
            [sys.executable, "main.py", "--cpu", "--listen", "127.0.0.1", "--port", str(PORT),
             "--disable-auto-launch", "--disable-all-custom-nodes", "--whitelist-custom-nodes", *packs,
             *(["--base-directory", str(args.base_directory)] if args.base_directory else [])],
            cwd=args.comfyui, stdout=log, stderr=subprocess.STDOUT,
        )
    failures = []
    try:
        deadline = time.monotonic() + args.timeout
        info = None
        while time.monotonic() < deadline and server.poll() is None:
            try:
                info = json.loads(get("/object_info")[1])
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                time.sleep(2)
        if info is None:
            failures.append("ComfyUI did not start (see log below)")
        else:
            log_text = log_path.read_text()
            for pack in packs:
                if any("IMPORT FAILED" in line and pack in line for line in log_text.splitlines()):
                    failures.append(f"{pack} reported IMPORT FAILED")
            for node in args.node:
                if node not in info:
                    failures.append(f"node type not registered: {node}")
            for route in args.route:
                try:
                    status, _ = get(route)
                except urllib.error.HTTPError as error:
                    status = error.code
                if status != 200:
                    failures.append(f"GET {route} answered {status}")
            for path in args.workflow:
                skipped = set()
                for node_type, cnr_id in workflow_types(path):
                    if cnr_id in ("comfy-core", args.cnr_id, *dependencies):
                        if node_type not in info:
                            failures.append(f"{path}: node type missing from ComfyUI: {node_type} ({cnr_id})")
                    elif cnr_id:
                        skipped.add(f"{node_type} ({cnr_id})")
                    else:
                        failures.append(f"{path}: node without a pack entry (cnr_id): {node_type}")
                for item in sorted(skipped):
                    print(f"{path}: skipped other pack's node {item}")
            print(f"Registered: {sorted(n for n in info if n in args.node)}")
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
    if failures:
        print(log_path.read_text()[-20000:])
        for failure in failures:
            print(f"::error::{failure}")
        return 1
    print("ComfyUI smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
