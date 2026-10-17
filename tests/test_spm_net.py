"""SPM-Net 多模态分割模型测试。

需要 torch；smp/transformers 缺失时对应用例会跳过。文本编码器用微型
HuggingFace 模型做真实集成测试（无网络时自动跳过）。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    HAS_TORCH = True
except Exception:  # ImportError 或损坏安装
    HAS_TORCH = False

if HAS_TORCH:
    import pytest
    import torch.nn as nn

    import dlkit.models  # noqa: F401
    from dlkit.models.zoo import spm_net as spm_module
    from dlkit.registry import MODELS


def _install_dummy_text_encoder(monkeypatch):
    class DummyTextEncoder(nn.Module):
        def __init__(self, model_name='dummy', output_dim=256, freeze_bert=True):
            super().__init__()
            self.freeze_bert = freeze_bert
            self.proj = nn.Linear(4, output_dim)

        def forward(self, text_descriptions, device):
            return torch.zeros(
                len(text_descriptions), self.proj.out_features, device=device,
            )

    monkeypatch.setattr(spm_module, 'TextEncoder', DummyTextEncoder)


if HAS_TORCH:

    def test_spm_net_registered():
        assert 'spm-net' in MODELS

    def test_spm_net_image_only():
        pytest.importorskip('segmentation_models_pytorch')
        model = spm_module.SPMNet(
            encoder_name='resnet18', encoder_weights=None,
            num_classes=4, use_text=False, use_structured=False,
        )
        model.eval()
        with torch.no_grad():
            logits = model(torch.randn(2, 3, 64, 64))
        assert logits.shape == (2, 4, 64, 64)

    def test_spm_net_multimodal_forward_and_backward(monkeypatch):
        pytest.importorskip('segmentation_models_pytorch')
        _install_dummy_text_encoder(monkeypatch)
        model = spm_module.SPMNet(
            encoder_name='resnet18', encoder_weights=None,
            num_classes=4, use_text=True, use_structured=True,
        )
        model.train()

        batch = {
            'image': torch.randn(2, 3, 64, 64),
            'immuno': torch.randn(2, 8),
            'clinical': torch.randn(2, 3),
            'text': ['description-a', 'description-b'],
        }
        logits = model.forward_batch(batch)
        assert logits.shape == (2, 4, 64, 64)

        logits.mean().backward()
        assert model.fusion_bottleneck.gate_generator[3].weight.grad is not None
        assert model.structured_encoder.fusion[0].weight.grad is not None

    def test_spm_net_accepts_legacy_batch_keys(monkeypatch):
        pytest.importorskip('segmentation_models_pytorch')
        _install_dummy_text_encoder(monkeypatch)
        model = spm_module.SPMNet(
            encoder_name='resnet18', encoder_weights=None,
            num_classes=4, use_text=True, use_structured=True,
        )
        model.eval()
        batch = {
            'image': torch.randn(2, 3, 64, 64),
            'immunohistochemistry': torch.randn(2, 8),
            'clinical_features': torch.randn(2, 3),
            'text_description': ['a', 'b'],
        }
        with torch.no_grad():
            logits = model.forward_batch(batch)
        assert logits.shape == (2, 4, 64, 64)

    def test_spm_net_checkpoint_keys_roundtrip(monkeypatch):
        pytest.importorskip('segmentation_models_pytorch')
        _install_dummy_text_encoder(monkeypatch)
        kwargs = dict(
            encoder_name='resnet18', encoder_weights=None,
            num_classes=4, use_text=True, use_structured=True,
        )
        first = spm_module.SPMNet(**kwargs)
        second = spm_module.SPMNet(**kwargs)
        second.load_state_dict(first.state_dict(), strict=True)
        for key, value in first.state_dict().items():
            assert torch.equal(value, second.state_dict()[key]), key

    def test_text_encoder_with_tiny_bert():
        pytest.importorskip('transformers')
        try:
            encoder = spm_module.TextEncoder(
                model_name='hf-internal-testing/tiny-random-bert',
                output_dim=16,
            )
        except Exception as exc:  # 无网络/无缓存时跳过
            pytest.skip('cannot load tiny bert model: %r' % (exc,))

        with torch.no_grad():
            features = encoder(['hello', 'world'], 'cpu')
        assert features.shape == (2, 16)
