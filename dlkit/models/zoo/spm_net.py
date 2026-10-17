"""SPM-Net：多模态分割网络（图像 + 结构化特征 + 文本）。

从原项目 V3 迁移而来：
- 保留模块结构与 state_dict 参数名，旧 checkpoint 可通过 ``train.init_from`` 直接加载；
- 用显式的 encoder -> fusion -> decoder -> head 前向替代 monkey-patching decoder；
- 依赖 ``segmentation-models-pytorch`` 与 ``transformers``（可选 extra）：

    uv sync --extra multimodal

训练 batch 约定（``forward_batch`` 会自动取以下键）::

    image:      (B, 3, H, W)
    immuno:     (B, 8)        # 别名 immunohistochemistry
    clinical:   (B, 3)        # 别名 clinical_features
    text:       list[str]     # 别名 text_description

说明：V3 中的 ``fusion_decoder`` 实际未被调用（原代码为 ``pass``）。本实现
保持同样的行为以保证指标可复现；该模块参数仍保留，以兼容旧 checkpoint。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dlkit.models.base import BaseModel
from dlkit.registry import MODELS


def _require_smp():
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:  # pragma: no cover - 环境缺依赖时的明确提示
        raise ImportError(
            'spm-net requires segmentation-models-pytorch; '
            'install with: uv sync --extra multimodal'
        ) from exc
    return smp


class ImprovedStructuredFeatureEncoder(nn.Module):
    """结构化特征编码器：学习免疫组化标记与临床特征的组合模式。"""

    def __init__(self, immuno_dim=8, clinical_dim=3, hidden_dim=128, output_dim=256):
        super().__init__()

        self.immuno_attention = nn.MultiheadAttention(
            embed_dim=immuno_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )
        self.immuno_encoder = nn.Sequential(
            nn.Linear(immuno_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, immuno_features, clinical_features):
        immuno_seq = immuno_features.unsqueeze(1)  # (B, 1, 8)
        attended, _ = self.immuno_attention(
            query=immuno_seq, key=immuno_seq, value=immuno_seq,
        )
        attended = attended.squeeze(1)

        immuno_encoded = self.immuno_encoder(attended)
        clinical_encoded = self.clinical_encoder(clinical_features)
        combined = torch.cat([immuno_encoded, clinical_encoded], dim=1)
        return self.fusion(combined)


class TextEncoder(nn.Module):
    """文本编码器：BERT CLS 特征 -> 投影。

    与 V3 的区别：冻结 BERT 时保持 eval 模式并使用 no_grad（原实现只在
    ``bert.training=False`` 时禁用梯度），避免训练时无谓的显存与计算开销。
    """

    def __init__(self, model_name='bert-base-chinese', output_dim=256, freeze_bert=True):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                'spm-net requires transformers; '
                'install with: uv sync --extra multimodal'
            ) from exc

        self.model_name = model_name
        self.freeze_bert = bool(freeze_bert)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        hidden_size = int(getattr(self.bert.config, 'hidden_size', 768))
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        if self.freeze_bert:
            for parameter in self.bert.parameters():
                parameter.requires_grad = False
            self.bert.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bert:
            self.bert.eval()
        return self

    def forward(self, text_descriptions, device):
        encoded = self.tokenizer(
            text_descriptions,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt',
        ).to(device)

        if self.freeze_bert:
            with torch.no_grad():
                outputs = self.bert(**encoded)
        else:
            outputs = self.bert(**encoded)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return self.projection(cls_embeddings)


class FiLMLayer(nn.Module):
    """FiLM 调制：用条件特征生成 gamma/beta，对图像特征做通道级仿射变换。"""

    def __init__(self, feature_dim, condition_dim):
        super().__init__()
        self.gamma_fc = nn.Linear(condition_dim, feature_dim)
        self.beta_fc = nn.Linear(condition_dim, feature_dim)

    def forward(self, x, condition):
        gamma = self.gamma_fc(condition).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_fc(condition).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta


class GatedMultimodalFusionModuleV3(nn.Module):
    """V3 门控多模态融合（AGF 2.0）。

    1. 瓶颈压缩 Gate：image_channels -> 256 -> image_channels；
    2. Gate 正向偏置初始化（sigmoid(2.0)≈0.88，初期保持开放）；
    3. 残差融合：``visual + gate * multimodal``，视觉特征永不丢失。
    """

    def __init__(self, image_channels=2048, structured_dim=256, text_dim=256,
                 output_channels=2048, num_classes=6):
        super().__init__()
        self.image_channels = image_channels
        self.num_classes = num_classes

        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, image_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.structured_film = FiLMLayer(image_channels, structured_dim)
        # 单向交叉注意力（Text -> Image）
        self.text_to_image_attention = nn.MultiheadAttention(
            embed_dim=image_channels,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        self.gate_generator = nn.Sequential(
            nn.Conv2d(image_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, image_channels, kernel_size=1),
        )
        nn.init.xavier_uniform_(self.gate_generator[3].weight)
        nn.init.constant_(self.gate_generator[3].bias, 2.0)

        self.final_fusion = nn.Sequential(
            nn.Conv2d(image_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, image_features, structured_features, text_features):
        batch, channels, height, width = image_features.shape

        text_proj = self.text_projection(text_features)
        image_modulated = self.structured_film(image_features, structured_features)

        image_seq = image_features.flatten(2).permute(0, 2, 1)
        text_seq = text_proj.unsqueeze(1)
        text_guided, _ = self.text_to_image_attention(
            query=image_seq, key=text_seq, value=text_seq,
        )
        text_guided = text_guided.permute(0, 2, 1).reshape(batch, channels, height, width)

        multimodal = image_modulated + text_guided
        gate = torch.sigmoid(self.gate_generator(image_features))
        fused = image_features + gate * multimodal
        return self.final_fusion(fused)


@MODELS.register(
    name='spm-net',
    tags=('segmentation', 'multimodal'),
    description='SPM-Net：图像 + 免疫组化/临床结构化特征 + 文本的多模态分割网络',
)
class SPMNet(BaseModel):
    """SPM-Net V3。

    参数与旧实现保持一致（encoder_name/encoder_weights/num_classes/
    use_text/use_structured/freeze_bert 等），新增的免疫组化/临床维度与
    BERT 模型名也做成了可配置项，默认值与旧代码相同。
    """

    _manual_build = True

    def __init__(
        self,
        encoder_name='resnet50',
        encoder_weights='imagenet',
        in_channels=3,
        num_classes=6,
        structured_dim=256,
        text_dim=256,
        use_text=True,
        use_structured=True,
        freeze_bert=True,
        immuno_dim=8,
        clinical_dim=3,
        text_model_name='bert-base-chinese',
    ):
        super().__init__()
        smp = _require_smp()

        self.use_text = use_text
        self.use_structured = use_structured
        self.num_classes = num_classes

        # 图像分割主干：smp UNet（旧 checkpoint 的 image_encoder.* 键名不变）
        self.image_encoder = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )

        encoder_channels = getattr(self.image_encoder.encoder, 'out_channels', None)
        bottleneck_channels = int(encoder_channels[-1]) if encoder_channels else 2048

        if use_structured:
            self.structured_encoder = ImprovedStructuredFeatureEncoder(
                immuno_dim=immuno_dim,
                clinical_dim=clinical_dim,
                hidden_dim=128,
                output_dim=structured_dim,
            )
        if use_text:
            self.text_encoder = TextEncoder(
                model_name=text_model_name,
                output_dim=text_dim,
                freeze_bert=freeze_bert,
            )

        if use_text or use_structured:
            self.fusion_bottleneck = GatedMultimodalFusionModuleV3(
                image_channels=bottleneck_channels,
                structured_dim=structured_dim,
                text_dim=text_dim,
                output_channels=bottleneck_channels,
                num_classes=num_classes,
            )
            # V3 定义但未使用；保留以兼容旧 checkpoint 的参数键
            self.fusion_decoder = FiLMLayer(512, structured_dim)

    def _encode_modalities(self, image, immunohistochemistry, clinical_features,
                           text_description):
        structured = None
        text = None
        if (self.use_structured and immunohistochemistry is not None
                and clinical_features is not None):
            structured = self.structured_encoder(immunohistochemistry, clinical_features)
        if self.use_text and text_description is not None:
            text = self.text_encoder(text_description, image.device)
        return structured, text

    def forward(self, image, immunohistochemistry=None, clinical_features=None,
                text_description=None):
        structured, text = self._encode_modalities(
            image, immunohistochemistry, clinical_features, text_description)

        features = self.image_encoder.encoder(image)
        if structured is not None and text is not None:
            features = list(features)
            features[-1] = self.fusion_bottleneck(features[-1], structured, text)
        decoder_output = self.image_encoder.decoder(features)
        return self.image_encoder.segmentation_head(decoder_output)

    def forward_batch(self, batch):
        """Trainer 的多模态入口：从 batch 中取各模态并前向。"""

        return self(
            batch['image'],
            batch.get('immuno', batch.get('immunohistochemistry')),
            batch.get('clinical', batch.get('clinical_features')),
            batch.get('text', batch.get('text_description')),
        )


__all__ = [
    'FiLMLayer',
    'GatedMultimodalFusionModuleV3',
    'ImprovedStructuredFeatureEncoder',
    'SPMNet',
    'TextEncoder',
]
