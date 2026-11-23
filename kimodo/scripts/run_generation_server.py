# SPDX-License-Identifier: Apache-2.0
"""HTTP server that generates a motion from text and returns it as a VMD for a given PMX model.

Used by mmd-kimodo (browser PMX/VMD editor). Only the prompt and the duration can be
chosen; every other parameter uses the Kimodo demo defaults (see ``DEMO_DEFAULTS``).

Endpoints:
    GET  /health    -> {"status": "loading" | "ready" | "error", "model": ..., "error": ...}
    POST /generate  multipart form: prompt (str), duration (seconds), pmx (file),
                    leg_ik (optional, default true) -> VMD bytes (application/octet-stream)

Example:
    python -m kimodo.scripts.run_generation_server --port 9560
"""

from __future__ import annotations

import argparse
import hashlib
import os
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

DEFAULT_SERVER_NAME = "0.0.0.0"
DEFAULT_SERVER_PORT = 9560

# Kimodo demo defaults (kimodo/demo/config.py and the Generate tab of kimodo/demo/ui.py).
DEMO_DEFAULTS = {
    "seed": 42,
    "diffusion_steps": 100,
    "cfg_type": "separated",
    "cfg_weight": [2.0, 2.0],  # text weight, constraint weight
    "num_transition_frames": 5,
    "post_processing": True,
    "root_margin": 0.04,
    "num_samples": 1,
}
MIN_DURATION = 2.0
MAX_DURATION = 10.0
MAX_PROMPT_LENGTH = 1000
MAX_PMX_BYTES = 256 * 1024 * 1024


@dataclass
class GeneratedMotion:
    joints_pos: np.ndarray  # [T, J, 3]
    global_rots: np.ndarray  # [T, J, 3, 3]
    fps: float


class KimodoGenerator:
    """Loads a Kimodo model once (in the background) and generates single text-to-motion samples."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.status = "loading"
        self.error: Optional[str] = None
        self._model = None
        self._lock = threading.Lock()  # one generation at a time (GPU)

    def start_loading(self) -> None:
        threading.Thread(target=self._load, name="kimodo-load", daemon=True).start()

    def _load(self) -> None:
        try:
            import torch

            from kimodo import DEFAULT_MODEL, load_model

            device = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            model, resolved = load_model(
                self.model_name or DEFAULT_MODEL,
                device=device,
                default_family="Kimodo",
                return_resolved_name=True,
            )
            self._model = model
            self.model_name = resolved
            self.device = device
            self.status = "ready"
            print(f"Kimodo model '{resolved}' loaded on {device}")
        except Exception as e:  # keep serving /health so the client can show the reason
            traceback.print_exc()
            self.status = "error"
            self.error = f"{type(e).__name__}: {e}"

    def generate(self, prompt: str, num_frames: int) -> GeneratedMotion:
        if self.status != "ready":
            raise RuntimeError(self.error or "Kimodo model is still loading")
        from kimodo.tools import seed_everything

        d = DEMO_DEFAULTS
        with self._lock:
            seed_everything(d["seed"])
            out = self._model(
                [prompt],
                [num_frames],
                d["diffusion_steps"],
                multi_prompt=True,
                constraint_lst=[],
                cfg_weight=list(d["cfg_weight"]),
                num_samples=d["num_samples"],
                cfg_type=d["cfg_type"],
                return_numpy=True,
                num_transition_frames=d["num_transition_frames"],
                post_processing=d["post_processing"] and "g1" not in (self.model_name or ""),
                root_margin=d["root_margin"],
            )
        joints_pos = np.asarray(out["posed_joints"])
        global_rots = np.asarray(out["global_rot_mats"])
        if joints_pos.ndim == 4:  # [B, T, J, 3]: single sample
            joints_pos, global_rots = joints_pos[0], global_rots[0]
        return GeneratedMotion(joints_pos, global_rots, float(self._model.fps))


class PmxCache:
    """Parsed PMX models by content hash (the same model is sent with every request)."""

    def __init__(self, size: int = 4, loader: Optional[Callable[[bytes], object]] = None):
        self.size = size
        self._loader = loader
        self._items: OrderedDict[str, object] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, data: bytes):
        key = hashlib.sha256(data).hexdigest()
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        if self._loader is None:
            from kimodo.mmd import load_pmx

            self._loader = load_pmx
        pmx = self._loader(data)
        with self._lock:
            self._items[key] = pmx
            while len(self._items) > self.size:
                self._items.popitem(last=False)
        return pmx


def default_to_vmd(motion: GeneratedMotion, pmx, leg_ik: bool) -> bytes:
    from kimodo.scripts.npz_to_vmd import motion_to_vmd_bytes

    payload, _ = motion_to_vmd_bytes(motion.joints_pos, motion.global_rots, pmx, fps=motion.fps, leg_ik=leg_ik)
    return payload


def create_app(generator, pmx_cache: Optional[PmxCache] = None, to_vmd=default_to_vmd):
    pmx_cache = pmx_cache or PmxCache()
    app = FastAPI(title="Kimodo generation server")

    @app.get("/health")
    def health():
        return {
            "status": generator.status,
            "model": generator.model_name,
            "error": generator.error,
            "defaults": DEMO_DEFAULTS,
            "minDuration": MIN_DURATION,
            "maxDuration": MAX_DURATION,
        }

    @app.post("/generate")
    def generate(
        prompt: str = Form(...),
        duration: float = Form(...),
        pmx: UploadFile = File(...),
        leg_ik: bool = Form(True),
    ):
        prompt = prompt.strip()
        if not prompt:
            raise HTTPException(400, "prompt is empty")
        if len(prompt) > MAX_PROMPT_LENGTH:
            raise HTTPException(400, f"prompt is longer than {MAX_PROMPT_LENGTH} characters")
        if not (MIN_DURATION <= duration <= MAX_DURATION):
            raise HTTPException(400, f"duration must be between {MIN_DURATION} and {MAX_DURATION} seconds")
        if generator.status == "loading":
            raise HTTPException(503, "Kimodo model is still loading")
        if generator.status != "ready":
            raise HTTPException(503, f"Kimodo model failed to load: {generator.error}")

        data = pmx.file.read(MAX_PMX_BYTES + 1)
        if len(data) > MAX_PMX_BYTES:
            raise HTTPException(413, "PMX file is too large")
        try:
            model = pmx_cache.get(data)
        except Exception as e:
            raise HTTPException(400, f"PMX parse error: {e}")

        fps = 30.0
        num_frames = int(round(duration * fps))
        t0 = time.perf_counter()
        try:
            motion = generator.generate(prompt, num_frames)
            payload = to_vmd(motion, model, leg_ik)
        except ValueError as e:  # e.g. skeleton has no MMD mapping
            raise HTTPException(422, str(e))
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(500, f"generation failed: {type(e).__name__}: {e}")
        elapsed = time.perf_counter() - t0
        print(f"Generated '{prompt}' ({num_frames} frames) in {elapsed:.1f}s")
        return Response(
            content=payload,
            media_type="application/octet-stream",
            headers={
                "X-Kimodo-Frames": str(int(motion.joints_pos.shape[0])),
                "X-Kimodo-Model": str(generator.model_name or ""),
                "X-Kimodo-Seconds": f"{elapsed:.2f}",
            },
        )

    return app


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run the Kimodo text-to-VMD generation server.")
    p.add_argument("--host", default=os.environ.get("SERVER_NAME", DEFAULT_SERVER_NAME))
    p.add_argument("--port", type=int, default=int(os.environ.get("GENERATION_SERVER_PORT", DEFAULT_SERVER_PORT)))
    p.add_argument("--model", default=os.environ.get("KIMODO_MODEL"), help="Model name (default: demo default)")
    p.add_argument("--device", default=None, help="Torch device (default: cuda:0 if available)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    import uvicorn

    args = parse_args(argv)
    generator = KimodoGenerator(args.model, args.device)
    generator.start_loading()
    uvicorn.run(create_app(generator), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
