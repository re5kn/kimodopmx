# SPDX-License-Identifier: Apache-2.0
"""Tests for kimodo.scripts.run_generation_server with a fake generator (no model / GPU needed).

Run: python -m pytest tests/test_generation_server.py
Needs fastapi, python-multipart, httpx and torch (for the skeleton assets).
Set MMD_TEST_PMX to a PMX with standard bone names to also test real retargeting.
"""

import os

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kimodo.scripts.run_generation_server import GeneratedMotion, PmxCache, create_app  # noqa: E402


class FakeGenerator:
    def __init__(self, status="ready", joints=77):
        self.status = status
        self.error = None if status != "error" else "boom"
        self.model_name = "fake"
        self.joints = joints
        self.calls = []

    def generate(self, prompt, num_frames):
        self.calls.append((prompt, num_frames))
        pos = np.zeros((num_frames, self.joints, 3), dtype=np.float32)
        rots = np.tile(np.eye(3, dtype=np.float32), (num_frames, self.joints, 1, 1))
        return GeneratedMotion(pos, rots, 30.0)


def _client(gen, to_vmd=None):
    cache = PmxCache(loader=lambda data: {"bytes": len(data)})
    kwargs = {"to_vmd": to_vmd} if to_vmd else {}
    return TestClient(create_app(gen, pmx_cache=cache, **kwargs))


def _post(client, prompt="A person walks forward.", duration="6.0", pmx=b"PMX fake"):
    return client.post(
        "/generate",
        data={"prompt": prompt, "duration": duration},
        files={"pmx": ("m.pmx", pmx, "application/octet-stream")},
    )


def test_health_reports_status_and_defaults():
    r = _client(FakeGenerator(status="loading")).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "loading"
    assert body["defaults"]["diffusion_steps"] == 100
    assert body["defaults"]["seed"] == 42


def test_generate_returns_vmd_from_converter():
    gen = FakeGenerator()
    seen = {}

    def to_vmd(motion, pmx, leg_ik):
        seen.update(frames=motion.joints_pos.shape[0], pmx=pmx, leg_ik=leg_ik)
        return b"Vocaloid Motion Data 0002"

    r = _post(_client(gen, to_vmd), duration="2.5")
    assert r.status_code == 200
    assert r.content == b"Vocaloid Motion Data 0002"
    assert r.headers["x-kimodo-frames"] == "75"
    assert gen.calls == [("A person walks forward.", 75)]
    assert seen == {"frames": 75, "pmx": {"bytes": 8}, "leg_ik": True}


@pytest.mark.parametrize(
    "prompt,duration",
    [("   ", "6"), ("x", "1.0"), ("x", "10.5"), ("x" * 1001, "6")],
)
def test_generate_rejects_bad_input(prompt, duration):
    gen = FakeGenerator()
    r = _post(_client(gen, lambda *a: b""), prompt=prompt, duration=duration)
    assert r.status_code == 400
    assert gen.calls == []


@pytest.mark.parametrize("status", ["loading", "error"])
def test_generate_unavailable_while_not_ready(status):
    r = _post(_client(FakeGenerator(status=status), lambda *a: b""))
    assert r.status_code == 503


def test_pmx_cache_parses_each_model_once():
    parsed = []
    cache = PmxCache(size=2, loader=lambda d: parsed.append(d) or d)
    for d in (b"a", b"a", b"b", b"a", b"c", b"b"):
        cache.get(d)
    # a, b parsed; a hit twice; c evicts b; b parsed again
    assert parsed == [b"a", b"b", b"c", b"b"]


@pytest.mark.skipif(not os.environ.get("MMD_TEST_PMX"), reason="set MMD_TEST_PMX to a PMX file")
def test_real_retarget_to_vmd():
    from kimodo.mmd import load_vmd

    with open(os.environ["MMD_TEST_PMX"], "rb") as f:
        pmx = f.read()
    client = TestClient(create_app(FakeGenerator(joints=77)))
    r = _post(client, duration="2", pmx=pmx)
    assert r.status_code == 200, r.text
    vmd = load_vmd(r.content)
    assert len(vmd.bone_frames) > 0
