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


import torch.nn as nn
import torch.nn.functional as F
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class VLM(EncoderDecoder):
    def __init__(self,
                 auxiliary_seg_head=None,
                 **args):
        super(VLM, self).__init__(**args)
        self._init_auxiliary_seg_head(auxiliary_seg_head)
            
    def _init_auxiliary_seg_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_seg_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_seg_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_seg_head = builder.build_head(auxiliary_head)

    def _auxiliary_seg_head_forward_test(self, x, need_contrast=False):
        if need_contrast:
            seg_logits, seg_fea = self.auxiliary_seg_head.forward(x, need_contrast)
            return seg_logits, seg_fea
        else:
            seg_logits = self.auxiliary_seg_head.forward(x, need_contrast)
            return seg_logits

    def freeze(self, model, exclude_keys=None):
        for n, m in model.named_parameters():
            m.requires_grad = False
            if exclude_keys is not None:
                assert isinstance(exclude_keys, list)
                for k in exclude_keys:
                    if str(k) in n:
                        m.requires_grad = True
                        print(f'Finetune {n}')

    def extract_feat(self, img):
        visual_feat = self.backbone(img)
        return visual_feat

    def _decode_head_forward_test(self, x, img_metas):
        # seg_logits = self.decode_head.forward(x, force_output_pred_masks=True)['pred_masks']
        seg_logits = self.decode_head.forward(x)
        return seg_logits
