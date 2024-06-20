"""
For instance,
python visualize.py --config exp/exp-47/xxx/config.yaml --checkpoint exp/exp-47/xxx/best.pth --save_dir tmp_vis

"""

import os
import argparse

import mmcv
import torch
import yaml
from torch.utils.data import DataLoader

from datasets.palettes import get_palette
from model.builder import build_model
from model.backbone.semi_resnet import semi_resnet50
from third_party.unimatch.dataset.semicd import SemiCDDataset
from utils.plot_utils import colorize_label

from version import __version__


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)

    args = parser.parse_args()

    with open(args.config, "r") as fp:
        cfg = yaml.load(fp, Loader=yaml.Loader)

    model = build_model(cfg)
    model.cuda()

    state_dict = torch.load(args.checkpoint)
    state_dict = state_dict['model']

    # multiple GPU to signle GPU
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)

    valset = SemiCDDataset(cfg, 'val')
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1)
    palette = get_palette(cfg['dataset'])

    os.makedirs(args.save_dir, exist_ok=True)

    for imgA_x, imgB_x, mask_x, id in valloader:
        imgA_x, imgB_x = imgA_x.cuda(), imgB_x.cuda()
        mask_x = mask_x.cuda()
        with torch.no_grad():
            model.eval()
            out = model(imgA_x, imgB_x)
            out = torch.argmax(out, dim=1)
            label = colorize_label(out.squeeze(0).cpu(), palette)
            mmcv.imwrite(label, os.path.join(args.save_dir, id[0]))
