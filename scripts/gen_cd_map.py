"""
Mixed Change Event Generation
"""

import os
import os.path as osp

import cv2
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    
    dir_A = 'APE_output/levir-cd_pseudo-label_ape_prob/A'
    dir_B = 'APE_output/levir-cd_pseudo-label_ape_prob/B'
    dst_dir = 'gen_cd_label/levir-cd_instance_mask-iou_0.0_direct_abs_ignore_0.8'
    os.makedirs(dst_dir, exist_ok=True)

    # load instance label
    instance_building_change_dir = 'gen_cd_label_vis/levir-cd_instance_mask-iou_0.0'
    threshold = 0.8

    for file_name in tqdm(os.listdir(dir_A)):
        if file_name.split('.')[-1] == 'npy':

            #### save seg label
            # pred_A = np.load(osp.join(dir_A, file_name))
            # pred_B = np.load(osp.join(dir_B, file_name))

            # pred_A = np.concatenate((pred_A, np.ones_like(pred_A[0:1, ...]) * 0.8), axis=0)
            # pred_B = np.concatenate((pred_B, np.ones_like(pred_B[0:1, ...]) * 0.8), axis=0)

            # pred_A = np.argmax(pred_A, axis=0)
            # pred_B = np.argmax(pred_B, axis=0)
            # pred_ignore_A = pred_A >= 6
            # pred_ignore_B = pred_B >= 6
            # pred_A[pred_A <= 1] = 1 # fg
            # pred_A[pred_A > 1] = 0 # bg
            # pred_B[pred_B <= 1] = 1 # fg
            # pred_B[pred_B > 1] = 0 # bg

            # pred_A[pred_ignore_A] = 255
            # pred_B[pred_ignore_B] = 255

            # cv2.imwrite(osp.join('gen_seg_label/levir-cd_direct_abs_ignore_0.8', 'A', \
            #                      file_name.split('.')[0]+'.png'), pred_A.astype('uint8'))
            # cv2.imwrite(osp.join('gen_seg_label/levir-cd_direct_abs_ignore_0.8', 'B', \
            #                      file_name.split('.')[0]+'.png'), pred_B.astype('uint8'))
            
            #### save cd label
            pred_A = np.load(osp.join(dir_A, file_name))
            pred_B = np.load(osp.join(dir_B, file_name))

            pred_A = np.concatenate((pred_A, np.ones_like(pred_A[0:1, ...]) * threshold), axis=0)
            pred_B = np.concatenate((pred_B, np.ones_like(pred_B[0:1, ...]) * threshold), axis=0)

            pred_A = np.argmax(pred_A, axis=0)
            pred_B = np.argmax(pred_B, axis=0)
            
            pred_ignore = (pred_A >= 6) | (pred_B >= 6)
            pred_A[pred_A <= 1] = 1 # bg
            pred_A[pred_A > 1] = 0 # fg
            pred_B[pred_B <= 1] = 1 # bg
            pred_B[pred_B > 1] = 0 # fg
            out = np.abs(pred_A - pred_B)
            # use instance label
            instance_building_change = cv2.imread(osp.join(instance_building_change_dir, file_name.split('.')[0] + '.png'), -1)
            instance_building_change = instance_building_change / 255
            out = out * instance_building_change
            out[pred_ignore] = 255
            cv2.imwrite(osp.join(dst_dir, file_name.split('.')[0]+'.png'), out.astype('uint8'))
