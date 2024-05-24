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


norm_cfg = dict(type='SyncBN', requires_grad=True)
img_size = 256

model = dict(
    type='VLM',
    backbone=dict(
        type='semi_resnet50',
        init_cfg=dict(type='Pretrained', checkpoint="pretrained/resnet50.pth"),
        replace_stride_with_dilation=[False, False, True]),
    decode_head=dict(
        type='LightHead',
        img_size=img_size,
        vl_sup=True,
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False),
    auxiliary_seg_head=dict(
        type='LightHead',
        img_size=img_size,
        vl_sup=False,
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False),
    # freeze_backbone=False,
    # exclude_keys=None,
)