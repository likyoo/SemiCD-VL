# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import types
from functools import reduce

import torch
from mmcv.utils import Config
from mmseg.models import build_segmentor
from mmseg.ops import resize
from torch.nn import functional as F

from model.backbone.timm_vit import TIMMVisionTransformer
from model.decode_heads.dlv3p_head import DLV3PHead
from model.backbone.semi_resnet import semi_resnet50
from model.decode_heads.light_head import LightHead
from model.vlm import VLM
from model.utils import DropPath
from third_party.unimatch.model.semseg.deeplabv3plus import DeepLabV3Plus


def nested_set(dic, key, value):
    keys = key.split('.')
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def nested_get(dictionary, keys, default=None):
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("."), dictionary)


def is_vlm(obj):
    return isinstance(obj, VLM)


def forward_wrapper(self, imgA, imgB, gt=None, need_fp=False, only_fp=False, forward_mode='default', need_seg_aux=False, need_contrast=False):
    # I'm sorry this part is written like a piece of shit, I'll try to optimize it in the future :(
    if forward_mode == 'default':
        xA = self.extract_feat(imgA)
        xB = self.extract_feat(imgB)

        x = []
        for f1, f2 in zip(xA, xB):
            x.append(torch.abs(f1 - f2))

        if self.disable_dropout:
            dropout_modules = [module for module in self.modules() if isinstance(module, torch.nn.Dropout) or isinstance(module, DropPath)]
            for module in dropout_modules:
                module.eval()
        if only_fp:
            x = [F.dropout2d(f, self.fp_rate) for f in x]
        elif need_fp:
            x = [torch.cat((f, F.dropout2d(f, self.fp_rate))) for f in x]
            xA = [torch.cat((f, F.dropout2d(f, self.fp_rate))) for f in xA]
            xB = [torch.cat((f, F.dropout2d(f, self.fp_rate))) for f in xB]

        out = self._decode_head_forward_test(x, img_metas=None)
        out_for_vl = None

        if isinstance(out, tuple):
            out, out_for_vl = out
            out = resize(
                input=out,
                size=imgA.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            out_for_vl = resize(
                input=out_for_vl,
                size=imgA.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        else:
            out = resize(
                input=out,
                size=imgA.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        
        if need_seg_aux:
            if need_contrast:
                outA, outA_fea = self._auxiliary_seg_head_forward_test(xA, need_contrast=need_contrast)
                outB, outB_fea = self._auxiliary_seg_head_forward_test(xB, need_contrast=need_contrast)
                dist = F.pairwise_distance(outA_fea, outB_fea, p=2.0, keepdim=True).permute(0, 3, 1, 2)
                dist = resize(
                    input=dist,
                    size=imgA.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
                dist = dist.squeeze(1)
            else:
                outA = self._auxiliary_seg_head_forward_test(xA, need_contrast=need_contrast)
                outB = self._auxiliary_seg_head_forward_test(xB, need_contrast=need_contrast)
            outA = resize(
                input=outA,
                size=imgA.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            outB = resize(
                input=outB,
                size=imgA.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            
        if need_fp:
            out = out.chunk(2)
            outA = outA.chunk(2)
            outB = outB.chunk(2)
            dist = dist.chunk(2)
            out_for_vl = out_for_vl.chunk(2)

        if need_seg_aux:
            if need_contrast:
                return out, out_for_vl, outA, outB, dist
            return out, out_for_vl, outA, outB
        return out
    else:
        raise ValueError(forward_mode)


def build_model(cfg):
    model_type = cfg['model']
    if model_type == 'deeplabv3plus':
        model = DeepLabV3Plus(cfg)
    elif 'mmseg.' in model_type:
        model_type = model_type.replace('mmseg.', '')
        model_cfg_file = f'configs/_base_/models/{model_type}.py'
        mmseg_cfg = Config.fromfile(model_cfg_file)
        mmseg_cfg['model']['decode_head']['num_classes'] = cfg['nclass']
        if 'zegclip' in model_type or 'vlm' in model_type:
            if mmseg_cfg['img_size'] != cfg['crop_size']:
                print('Modify model image_size to match crop_size', cfg['crop_size'])
                nested_set(mmseg_cfg, 'img_size', cfg['crop_size'])
                nested_set(mmseg_cfg, 'model.backbone.img_size', (cfg['crop_size'],  cfg['crop_size']))
                nested_set(mmseg_cfg, 'model.decode_head.img_size', cfg['crop_size'])
        if 'model_args' in cfg:
            mmseg_cfg['model'].update(cfg['model_args'])
        model = build_segmentor(
            mmseg_cfg.model,
            train_cfg=mmseg_cfg.get('train_cfg'),
            test_cfg=mmseg_cfg.get('test_cfg'))
        model.disable_dropout = cfg['disable_dropout']
        model.fp_rate = cfg['fp_rate']
        model.forward = types.MethodType(forward_wrapper, model)
        model.init_weights()
    else:
        raise ValueError(model_type)
    
    return model
