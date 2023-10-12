import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms


class DecodeBox(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(-1)           # [B, 100, 4] -> 4 * [B, 100]
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),  # 4 * [B, 100] -> [[B, 100] * 4]
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)           # [[B, 100] * 4] -> [B, 100, 4]

    @torch.no_grad()
    def forward(self, outputs, target_sizes, confidence):
        """
        Args:
            outputs (dict): {'pred_logits': [B, 100, num_classes+1], 'pred_boxes': [B, 100, 4]}
            target_sizes (Tensor): [H, W]
            confidence (float): _description_

        Returns:
            list[Tensor]:
        """
        # out_logits: [B, 100, num_classes+1]
        # pred_boxes: [B, 100, 4]
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # 调整概率 [B, 100, num_classes+1] -> [B, 100, num_classes+1]
        prob = F.softmax(out_logits, -1)
        # 忽略最后的背景类别
        # 得到最高分和id [B, 100] * 2
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = self.box_cxcywh_to_xyxy(out_bbox)       # [B, 100, 4] -> [B, 100, 4]
        # 还原到原图尺寸,模型中进行了sigmoid
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w    = target_sizes.unbind(1)
        img_h           = img_h.float()
        img_w           = img_w.float()
        scale_fct       = torch.stack([img_w, img_h, img_w, img_h], dim=1)  # [1, 4]
        boxes           = boxes * scale_fct[:, None, :] # [B, 100, 4] * ([1, 4] -> [1, 1, 4]) = [B, 100, 4]

        # [B, 100, 6]   6: y1 x1 y2 x2 score label
        outputs = torch.cat([
                torch.unsqueeze(boxes[:, :, 1], -1),
                torch.unsqueeze(boxes[:, :, 0], -1),
                torch.unsqueeze(boxes[:, :, 3], -1),
                torch.unsqueeze(boxes[:, :, 2], -1),
                torch.unsqueeze(scores, -1),
                torch.unsqueeze(labels.float(), -1),
            ], -1)

        results = []
        for output in outputs:
            # [100, 6] get [keep, 6]
            results.append(output[output[:, 4] > confidence])

        # [[keep, 6]]   6: y1 x1 y2 x2 score label
        # results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results
