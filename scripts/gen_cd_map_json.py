"""
Instance-level Change Event Generation
"""

import os
import os.path as osp

import cv2
import json
import tqdm
import numpy as np
import pycocotools.mask as maskUtils


src_dir = 'APE_output/levir-cd_pseudo-label_ape_prob'
dst_dir = 'gen_cd_label/levir-cd_instance_mask-iou_0.0'
os.makedirs(dst_dir, exist_ok=True)

for file_name in tqdm.tqdm(os.listdir(osp.join(src_dir, 'A'))):
    if file_name.split('.')[-1] == 'json':

        annotation_file_A = osp.join(src_dir, 'A', file_name)
        annotation_file_B = osp.join(src_dir, 'B', file_name)

        change_instances = []
        iscrowd = [0]
        seg_mask = np.zeros((256, 256)).astype(bool)

        with open(annotation_file_A, 'r') as f:
            datasetA = json.load(f)

        with open(annotation_file_B, 'r') as f:
            datasetB = json.load(f)

        g = [g['segmentation'] for g in datasetA if g['category_id'] in [0, 1]]
        d = [d['segmentation'] for d in datasetB if d['category_id'] in [0, 1]]

        if len(g) == 0 and len(d) != 0:
            change_instances = d
        elif len(g) != 0 and len(d) == 0:
            change_instances = g
        elif len(g) == 0 and len(d) == 0:
            cv2.imwrite(osp.join(dst_dir, file_name.split('.')[0] + '.png'), seg_mask.astype('uint8'))
            continue
        else:
            ious = maskUtils.iou(d, g, iscrowd * len(g))
            ious_g = ious.sum(0)
            ious_d = ious.sum(1)
            change_g = list(np.where(ious_g <= 0)[0])
            change_d = list(np.where(ious_d <= 0)[0])
            change_instances = [g[i] for i in change_g] +  [d[i] for i in change_d]

        for ins in change_instances:
            m = maskUtils.decode(ins).astype(bool)
            seg_mask = (seg_mask | m)

        cv2.imwrite(osp.join(dst_dir, file_name.split('.')[0] + '.png'), (seg_mask*255).astype('uint8'))
