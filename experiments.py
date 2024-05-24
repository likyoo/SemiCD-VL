from argparse import ArgumentParser
from functools import reduce
import itertools
import yaml
import os
import os.path as osp
import subprocess
import collections.abc
from version import __version__


DATA_DIR = './data/'

def nested_set(dic, key, value):
    keys = key.split('.')
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value

def nested_get(dictionary, keys, default=None):
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("."), dictionary)

def nested_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = nested_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def get_git_revision() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except subprocess.CalledProcessError:
        return ''

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

def config_from_vars(
    exp_id,
    gpu_model='4090',
    n_gpus=1,
    n_nodes=1,
    batch_size=8,
    epochs=80,
    iters=None,
    scheduler_max_iters=None,
    dataset='levir',
    split='5%',
    img_scale=[256, 256],
    scale_ratio_range=(0.5, 2.0),
    crop_size=256,
    labeled_photometric_distortion=False,
    method='diffmatch',
    use_fp=True,
    conf_mode='pixelwise',
    conf_thresh=0.95,
    pleval=True,
    disable_dropout=True,
    fp_rate=0.5,
    vl_consistency_lambda=0,
    vl_loss_reduce='mean',
    opt='adamw',
    lr=1e-4,
    backbone_lr_mult=10.0,
    conv_enc_lr_mult=1.0,
    warmup_iters=0,
    criterion='mmseg',
    criterion_u='mmseg',
    model='mmseg.zegclip-vitb',
    eval_mode='original',
    eval_every=1,
    nccl_p2p_disable=False,
    vl_label_root=None,
    vl_change_label_root=None,
    contrastive_loss_weight=0.1,
):
    cfg = dict()
    name = ''

    # Dataset
    cfg['dataset'] = dataset
    name += dataset.replace('pascal', 'voc').replace('cityscapes', 'cs')
    cfg['data_root'] = dict(
        levir=osp.join(DATA_DIR, 'LEVIR-CD-256/'),
        whu=osp.join(DATA_DIR, 'WHU-CD-256/'),
    )[dataset]
    cfg['nclass'] = dict(
        levir=2,
        whu=2,
    )[dataset]
    cfg['split'] = split
    name += f'-{split}'
    cfg['img_scale'] = img_scale
    if img_scale is not None:
        name += f'-{img_scale}'
    cfg['scale_ratio_range'] = scale_ratio_range
    if scale_ratio_range != (0.5, 2.0):
        name += f'-s{scale_ratio_range[0]}-{scale_ratio_range[1]}'
    cfg['crop_size'] = crop_size
    name += f'-{crop_size}'
    cfg['labeled_photometric_distortion'] = labeled_photometric_distortion
    if labeled_photometric_distortion:
        name += '-phd'

    # Model
    name += f'_{model}'.replace('mmseg.', '').replace('zegclip', 'zcl')
    cfg['model_args'] = {}
    if model == 'dlv3p-r101':
        cfg['model'] = 'deeplabv3plus'
        cfg['backbone'] = 'resnet101'
        cfg['replace_stride_with_dilation'] = [False, False, True]
        cfg['dilations'] = [6, 12, 18]
    elif model == 'dlv3p-r50':
        cfg['model'] = 'deeplabv3plus'
        cfg['backbone'] = 'resnet50'
        cfg['replace_stride_with_dilation'] = [False, False, True]
        cfg['dilations'] = [6, 12, 18]
    elif model == 'dlv3p-xc65':
        cfg['model'] = 'deeplabv3plus'
        cfg['backbone'] = 'xception'
        cfg['dilations'] = [6, 12, 18]
    else: # mmseg.
        cfg['model'] = model

    # Method
    cfg['method'] = method
    name += f'_{method}'.replace('diffmatch', 'dm').replace('supervised', 'sup')
    if method in ['diffmatch_unimatch', 'diffmatch_fixmatch']:
        cfg['use_fp'] = use_fp
        if not use_fp:
            name += '-nfp'
        cfg['conf_mode'] = conf_mode
        name += {
            'pixelwise': '',
            'pixelratio': '-cpr',
            'pixelavg': '-cpa',
        }[conf_mode]
        cfg['conf_thresh'] = conf_thresh
        name += f'-{conf_thresh}'
    cfg['disable_dropout'] = disable_dropout
    if disable_dropout:
        name += '-disdrop'
    if method in ['diffmatch', 'diffmatch_unimatch', 'diffmatch_fixmatch']:
        cfg['pleval'] = pleval
        if pleval:
            name += '-plev'
    cfg['fp_rate'] = fp_rate
    if fp_rate != 0.5:
        name += f'-fpr{fp_rate}'
    cfg['vl_consistency_lambda'] = vl_consistency_lambda
    cfg['vl_loss_reduce'] = vl_loss_reduce
    name += {
        'mean': '',
        'mean_valid': '-mv',
        'mean_all': '-ma',
    }[vl_loss_reduce]

    # Criterion
    cfg['criterion'] = dict(
        name=criterion,
        kwargs=dict(ignore_index=255)
    )
    if cfg['criterion'] == 'OHEM':
        cfg['criterion']['kwargs'].update(dict(
            thresh=0.7,
            min_kept=200000
        ))
    if criterion != 'mmseg':
        name += f'-{criterion}'.replace('CELoss', 'ce').replace('OHEM', 'oh')
    cfg['criterion_u'] = criterion_u
    if criterion_u != 'mmseg':
        name += f'-u{criterion_u}'.replace('CELoss', 'ce')

    # Optimizer
    if opt == 'original':
        cfg['lr'] = lr
        # cfg['lr_multi'] = 10.0 if dataset != 'cityscapes' else 1.0
        cfg['lr_multi'] = backbone_lr_mult
    elif opt == 'adamw':
        cfg['optimizer'] = dict(
            type='AdamW', lr=lr, weight_decay=0.01,
            paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=backbone_lr_mult),
                                            'text_encoder': dict(lr_mult=0.0),
                                            'conv_encoder': dict(lr_mult=conv_enc_lr_mult),
                                            'norm': dict(decay_mult=0.),
                                            'ln': dict(decay_mult=0.),
                                            'head': dict(lr_mult=10.),
                                            }))
    else:
        raise NotImplementedError(opt)
    name += f'_{opt}-{lr:.0e}'.replace('original', 'org')
    if backbone_lr_mult != 10.0:
        name += f'-b{backbone_lr_mult}'
    if conv_enc_lr_mult != 1.0:
        name += f'-cl{conv_enc_lr_mult}'
    cfg['warmup_iters'] = warmup_iters
    cfg['warmup_ratio'] = 1e-6
    if warmup_iters > 0:
        name += f'-w{human_format(warmup_iters)}'

    # Batch
    cfg['gpu_model'] = gpu_model
    cfg['n_gpus'] = n_gpus
    cfg['n_nodes'] = n_nodes
    cfg['batch_size'] = batch_size
    if n_gpus != 4 or batch_size != 2 or n_nodes != 1:
        name += f'_{n_nodes}x{n_gpus}x{batch_size}'

    # Schedule
    assert not (iters is not None and epochs is not None)
    cfg['epochs'] = epochs
    cfg['iters'] = iters
    if epochs is not None and epochs != 80:
        name += f'-ep{human_format(epochs)}'
    if iters is not None:
        name += f'-i{human_format(iters)}'
    if scheduler_max_iters is not None:
        cfg['scheduler_max_iters'] = scheduler_max_iters
        name += f'-smi{scheduler_max_iters}'

    # Eval
    cfg['eval_mode'] = eval_mode
    if eval_mode == 'zegclip_sliding_window':
        cfg['stride'] = 256 # 426
    name += '_e' + {
        'original': 'or',
        'sliding_window': 'sw',
        'zegclip_sliding_window': 'zsw',
    }[eval_mode]
    cfg['eval_every_n_epochs'] = eval_every
    cfg['nccl_p2p_disable'] = nccl_p2p_disable

    # diffmatch
    cfg['vl_label_root'] = vl_label_root
    cfg['vl_change_label_root'] = vl_change_label_root
    cfg['contrastive_loss_weight'] = contrastive_loss_weight

    cfg['exp'] = exp_id
    cfg['name'] = name.replace('.0_', '').replace('.0-', '').replace('.', '').replace('True', 'T')\
        .replace('False', 'F').replace('None', 'N').replace('[', '')\
        .replace(']', '').replace('(', '').replace(')', '').replace(',', 'j')\
        .replace(' ', '')
    cfg['version'] = __version__
    cfg['git_rev'] = get_git_revision()

    return cfg

def generate_experiment_cfgs(exp_id):
    cfgs = []
    # -------------------------------------------------------------------------
    # DiffMatch for WHU-CD-256
    # -------------------------------------------------------------------------
    if exp_id == 48:
        n_repeat = 1
        splits = ['5%', '10%', '20%', '40%']
        list_kwargs = [
            dict(model='mmseg.vlm-lighthead-r50', opt='original', lr=2e-2, backbone_lr_mult=10., criterion='CELoss',
                 vl_consistency_lambda=[0.1, 0], vl_loss_reduce='mean_all',
                 vl_label_root='gen_seg_label/whu-cd_direct_abs_ignore_0.8', 
                 vl_change_label_root='gen_cd_label/whu-cd_instance_mask-iou_0.0_direct_abs_ignore_0.8',
                 contrastive_loss_weight=0.1)]
        for split, kwargs, _ in itertools.product(splits, list_kwargs, range(n_repeat)):
            cfg = config_from_vars(
                exp_id=exp_id,
                dataset='whu',
                method='diffmatch_fixmatch',
                split=str(split),
                conf_thresh=0.95,
                crop_size=256,
                img_scale=None,
                eval_mode='original',
                criterion_u='CELoss',
                **kwargs,
            )
            cfgs.append(cfg)

    # -------------------------------------------------------------------------
    # DiffMatch for LEVIR-CD-256
    # -------------------------------------------------------------------------
    elif exp_id == 47:
        n_repeat = 1
        splits = ['5%', '10%', '20%', '40%']
        list_kwargs = [
            dict(model='mmseg.vlm-lighthead-r50', opt='original', lr=2e-2, backbone_lr_mult=10., criterion='CELoss',
                 vl_consistency_lambda=[0.1, 0], vl_loss_reduce='mean_all',
                 vl_label_root='gen_seg_label/levir-cd_direct_abs_ignore_0.8', 
                 vl_change_label_root='gen_cd_label/levir-cd_instance_mask-iou_0.0_direct_abs_ignore_0.8',
                 contrastive_loss_weight=0.1)]
        for split, kwargs, _ in itertools.product(splits, list_kwargs, range(n_repeat)):
            cfg = config_from_vars(
                exp_id=exp_id,
                dataset='levir',
                method='diffmatch_fixmatch',
                split=str(split),
                conf_thresh=0.95,
                crop_size=256,
                img_scale=None,
                eval_mode='original',
                criterion_u='CELoss',
                **kwargs,
            )
            cfgs.append(cfg)
    else:
        raise NotImplementedError(f'Unknown id {exp_id}')

    return cfgs


def save_experiment_cfgs(exp_id):
    cfgs = generate_experiment_cfgs(exp_id)
    cfg_files = []
    for cfg in cfgs:
        cfg_file = f"configs/generated/exp-{cfg['exp']}/{cfg['name']}.yaml"
        os.makedirs(os.path.dirname(cfg_file), exist_ok=True)
        with open(cfg_file, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=None, sort_keys=False, indent=2)
        cfg_files.append(cfg_file)

    return cfgs, cfg_files

def run_command(command):
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    for line in iter(p.stdout.readline, b''):
        print(line.decode('utf-8'), end='')

if __name__ == '__main__':
    parser = ArgumentParser(description='Generate experiment configs')
    parser.add_argument('--exp', type=int, help='Experiment id')
    parser.add_argument('--run', type=int, default=0, help='Run id')
    parser.add_argument('--ngpus', type=int, default=None, help='Override number of GPUs')
    args = parser.parse_args()

    cfgs, cfg_files = save_experiment_cfgs(args.exp)

    if args.ngpus is None:
        ngpus = cfgs[args.run]["n_gpus"]
    else:
        ngpus = args.ngpus

    cmd = f'bash scripts/train.sh {cfgs[args.run]["method"]} {cfg_files[args.run]} {ngpus}'
    print(cmd)
    run_command(cmd)
