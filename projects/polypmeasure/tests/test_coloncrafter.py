import numpy as np
import pytest
import torch

from dlkit.builders import build_from_cfg
from polypmeasure.bootstrap import register


def _fake_implementation_factory():
    class FakeImplementation:
        @classmethod
        def from_pretrained(cls, path, device="cuda"):
            return cls()

        def predict_depth(self, video, **kwargs):
            frame_count = video.shape[0]
            height, width = video.shape[-2:]
            depth = np.ones((frame_count, height, width), dtype=np.float32) * 2.0
            disparity = np.ones((frame_count, height, width), dtype=np.float32) * 0.5
            return depth, disparity

    return FakeImplementation


def _build(**params):
    register()
    return build_from_cfg(
        {
            "type": "ColonCrafter",
            "params": {"model_path": "fake/repo", **params},
        }
    )


def test_build_and_forward_with_fake_implementation(monkeypatch):
    from polypmeasure.components.models import coloncrafter

    monkeypatch.setattr(
        coloncrafter,
        "_import_coloncrafter",
        lambda directory: _fake_implementation_factory(),
    )
    model = _build(coloncrafter_dir="/fake/dir", device="cpu")

    output = model(torch.rand(3, 3, 32, 32), num_inference_steps=1)

    assert output.shape == (3, 32, 32)
    assert float(output.mean()) == 2.0


def test_missing_dependency_gives_guidance():
    model = _build(device="cpu")

    with pytest.raises(RuntimeError, match="git clone"):
        model(torch.rand(2, 3, 32, 32))


def test_nonexistent_checkout_gives_guidance(tmp_path):
    missing = tmp_path / "missing-coloncrafter"
    model = _build(coloncrafter_dir=missing, device="cpu")

    with pytest.raises(RuntimeError, match="不存在"):
        model(torch.rand(2, 3, 32, 32))
