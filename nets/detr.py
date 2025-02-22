# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn.functional as F
from torch import nn

from . import ops
from .backbone import build_backbone, FrozenBatchNorm2d
from .ops import NestedTensor, nested_tensor_from_tensor_list, unused
from .transformer import build_transformer



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class DETR(nn.Module):
    def __init__(self, backbone, position_embedding, hidden_dim, num_classes, num_queries, aux_loss=False, pretrained=False):
        super().__init__()

        #-------------------------------------------------------------------#
        #   传入主干网络中进行预测
        #   [B, 3, 800, 800] => [[B, 2048, 25, 25]], [[B, 256, 25, 25]]
        #-------------------------------------------------------------------#
        self.backbone       = build_backbone(backbone, position_embedding, hidden_dim, pretrained=pretrained)
        #-------------------------------------------------------------------#
        #   将主干的结果进行一个映射，然后和查询向量和位置向量传入transformer。
        #   [B, 2048, 25, 25] => [B, 256, 25, 25]
        #-------------------------------------------------------------------#
        self.input_proj     = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)

        #----------------------#
        #   Encoder + Decoder
        #----------------------#
        self.transformer    = build_transformer(hidden_dim=hidden_dim, pre_norm=False)
        hidden_dim          = self.transformer.d_model

        #------------#
        #   Header
        #------------#
        # 输出分类信息
        self.class_embed    = nn.Linear(hidden_dim, num_classes + 1)
        # 输出回归信息
        self.bbox_embed     = MLP(hidden_dim, hidden_dim, 4, 3)
        # 用于传入transformer进行查询的查询向量 [100, 256]
        self.query_embed    = nn.Embedding(num_queries, hidden_dim)

        # 查询向量的长度与是否使用辅助分支
        self.num_queries    = num_queries
        self.aux_loss       = aux_loss

    def forward(self, samples: NestedTensor):
        # samples: [B, 3, 800, 800]
        if isinstance(samples, (list, torch.Tensor)):
            # [B, 3, 800, 800] => [B, 3, 800, 800], [B, 800, 800]
            samples = nested_tensor_from_tensor_list(samples)

        #-------------------------------------------------------------------#
        #   传入主干网络中进行预测
        #   tensors,          mask             features               pos
        #   [B, 3, 800, 800], [B, 800, 800] => [[B, 2048, 25, 25]], [[B, 256, 25, 25]] 注意返回值都为列表,里面有1个tensor
        #-------------------------------------------------------------------#
        features, pos = self.backbone(samples)

        #----------------------#
        #   Encoder + Decoder
        #----------------------#
        # 将网络的结果进行分割，把特征和mask进行分开
        # features[-1]为NestedTensor,里面存放tensors和mask,使用 decompose() 返回2个值
        # [B, 2048, 25, 25] => [B, 2048, 25, 25], [B, 25, 25]
        src, mask = features[-1].decompose()
        assert mask is not None
        # 将主干的结果进行一个映射，然后和查询向量和位置向量传入transformer。
        # [B, 2048, 25, 25] => [B, 256, 25, 25]
        src = self.input_proj(src)
        #--------------------------------------------------------#
        #   src:          [B, 256, 25, 25]
        #   mask:         [B, 25, 25]
        #   query_embed:  [100, 256]
        #   pos_embed:    [B, 256, 25, 25] => [6, B, 100, 256]  6指的是TransformerDecoderLayer的6个layer的输出
        #   transformer返回2个值,只要第1个
        #--------------------------------------------------------#
        hs = self.transformer(src, mask, self.query_embed.weight, pos[-1])[0]

        #------------#
        #   Header
        #------------#
        # 输出分类信息 1个Linear
        # [6, B, 100, 256] => [6, B, 100, num_classes+1]
        outputs_class = self.class_embed(hs)
        # 输出回归信息,调整到0~1之间
        # [6, B, 100, 256] => [6, B, 100, 4]
        outputs_coord = self.bbox_embed(hs).sigmoid()

        # 只输出transformer最后一层的内容
        # [B, 100, num_classes+1], [B, 100, 4]
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss: # 输出transformer前几层的内容
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, FrozenBatchNorm2d):
                m.eval()


if __name__ == "__main__":
    detr = DETR(
        backbone="resnet50",
        position_embedding="sine",
        hidden_dim=256,
        num_classes=20,
        num_queries=100,
    )
    detr.eval()

    x = torch.ones(1, 3, 800, 800)
    with torch.inference_mode():
        y = detr(x)
    print(y["pred_logits"].shape)   # [B, 100, 21]
    print(y["pred_boxes"].shape)    # [B, 100, 4]
