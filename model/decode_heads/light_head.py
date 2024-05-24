import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize


@HEADS.register_module()
class LightHead(BaseDecodeHead):
    """
    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, img_size, vl_sup=False, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.img_size = img_size
        self.vl_sup = vl_sup
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        
        if self.vl_sup:
            self.conv_seg_for_vl = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)

    def forward(self, inputs, need_contrast=False):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out_fea = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.cls_seg(out_fea)

        if need_contrast:
            return out, out_fea

        if self.vl_sup:
            out_for_vl = self.conv_seg_for_vl(out_fea)
            return (out, out_for_vl)

        return out
