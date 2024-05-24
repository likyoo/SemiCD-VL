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

import argparse
import logging
import math
import os
import pprint
import shutil
import uuid
import time
from datetime import datetime

import mmcv
import torch
import torch.backends.cudnn as cudnn
import yaml
from matplotlib import pyplot as plt
from mmseg.core import build_optimizer
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.palettes import get_palette
from experiments import get_git_revision
from model.builder import build_model
from model.utils import BCLLoss
from third_party.unimatch.supervised import evaluate
from third_party.unimatch.dataset.semicd import SemiCDDataset
from datasets.classes import CLASSES
from third_party.unimatch.util.ohem import ProbOhemCrossEntropy2d
from third_party.unimatch.util.dist_helper import setup_distributed
from third_party.unimatch.util.utils import count_params, count_training_params, init_log
from utils.gen_code_archive import gen_code_archive
from utils.plot_utils import plot_data, colorize_label
from utils.train_utils import (DictAverageMeter, confidence_weighted_loss,
                               cutmix_img_, cutmix_mask)
from version import __version__


def compute_vl_loss(pred, mask, ign):
    l_mc = criterion_mc(pred, mask)
    if vl_loss_reduce == 'mean_valid':
        l_mc = l_mc.sum() / (ign != 255).sum()
    if vl_loss_reduce == 'mean_all':
        l_mc = l_mc.sum() / ign.numel()
    return l_mc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--port', default=None, type=int)

    args = parser.parse_args()

    with open(args.config, "r") as fp:
        cfg = yaml.load(fp, Loader=yaml.Loader)
    labeled_id_path = f'splits/{cfg["dataset"]}/{cfg["split"]}/labeled.txt'
    unlabeled_id_path = f'splits/{cfg["dataset"]}/{cfg["split"]}/unlabeled.txt'

    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    mmcv.utils.get_logger('mmcv').setLevel('WARNING')

    rank, world_size = setup_distributed(port=args.port)
    if cfg['nccl_p2p_disable']:
        os.environ["NCCL_P2P_DISABLE"] = str(1)

    if rank == 0:
        timestr = datetime.now().strftime("%y%m%d-%H%M")
        uid = str(uuid.uuid4())[:5]
        run_name = f'{timestr}_{cfg["name"]}_v{__version__}_{uid}'.replace('.', '-')
        save_path = f'exp/exp-{cfg["exp"]}/{run_name}'
        os.makedirs(save_path, exist_ok=True)

        formatter = logging.Formatter(fmt='[%(asctime)s] [%(levelname)-8s] %(message)s')
        fileHandler = logging.FileHandler(f'{save_path}/debug.log')
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

        all_args = {**cfg, **vars(args), 
                    'labeled_id_path': labeled_id_path, 'unlabeled_id_path': unlabeled_id_path,
                    'ngpus': world_size, 'run_name': run_name, 'save_path': save_path,
                    'exec_git_rev': get_git_revision(), 'exec_version': __version__}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(save_path)
        
        shutil.copyfile(args.config, os.path.join(save_path, 'config.yaml'))
        with open(os.path.join(save_path, 'all_args.yaml'), 'w') as f:
            yaml.dump(all_args, f, default_flow_style=None, sort_keys=False, indent=2)
        gen_code_archive(save_path)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    vl_consistency_lambda = cfg['vl_consistency_lambda']
    vl_loss_reduce = cfg['vl_loss_reduce']
    assert vl_loss_reduce in ['mean', 'mean_valid', 'mean_all']
    assert cfg['use_fp']
    assert cfg['pleval']

    contrastive_loss_weight = cfg['contrastive_loss_weight']

    model = build_model(cfg)
    if 'optimizer' not in cfg:
        optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                        {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                        'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = build_optimizer(model, cfg['optimizer'])
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
    
    if rank == 0:
        logger.info(model)
        logger.info(f'Total params: {count_params(model):.1f}M\n')
        if hasattr(model, 'backbone'):
            logger.info(f'Backbone params (training/total): {count_training_params(model.backbone):.1f}M/{count_params(model.backbone):.1f}M\n')
        if hasattr(model, 'decode_head'):
            logger.info(f'Decoder params (training/total): {count_training_params(model.decode_head):.1f}M/{count_params(model.decode_head):.1f}M\n')

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=True)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'mmseg':
        criterion_l = None
    else:
        raise ValueError(cfg['criterion_u']['name'])
    
    criterion_dist = BCLLoss(margin=2.0, loss_weight=1.0, ignore_index=255)

    if cfg['criterion_u'] == 'CELoss':
        criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)
    elif cfg['criterion_u'] == 'mmseg':
        criterion_u = None
    else:
        raise ValueError(cfg['criterion_u'])

    if vl_consistency_lambda != 0:
        if vl_loss_reduce == 'mean':
            criterion_mc = nn.CrossEntropyLoss(ignore_index=255).cuda(local_rank)
        elif vl_loss_reduce in ['mean_valid', 'mean_all']:
            criterion_mc = nn.CrossEntropyLoss(ignore_index=255, reduction='none').cuda(local_rank)
        else:
            raise ValueError(vl_loss_reduce)

    trainset_u = SemiCDDataset(cfg, 'train_u', id_path=unlabeled_id_path)
    trainset_l = SemiCDDataset(cfg, 'train_l', id_path=labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiCDDataset(cfg, 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)
    palette = get_palette(cfg['dataset'])

    if cfg['iters'] is not None:
        assert cfg['epochs'] is None
        cfg['epochs'] = math.ceil(cfg['iters'] / len(trainloader_u))

    total_iters = len(trainloader_u) * cfg['epochs']
    scheduler_max_iters = cfg.get('scheduler_max_iters', total_iters)
    assert scheduler_max_iters >= total_iters
    if rank == 0:
        logger.info(f'Train for {cfg["epochs"]} epochs / {total_iters} iterations.')
    previous_best_iou, previous_best_acc = 0.0, 0.0
    epoch = -1
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best Changed IoU: {:.2f}, Overall Accuracy: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best_iou, previous_best_acc))

        log_avg = DictAverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        for i, ((imgA_x, imgB_x, mask_x, vl_mask_x_change, vl_mask_x_A, vl_mask_x_B),
                (imgA_w, imgB_w, imgA_s1, imgB_s1,
                 _, _, ignore_mask, mix1, _, vl_mask_change, vl_mask_A, vl_mask_B),
                (imgA_w_other, imgB_w_other, imgA_s1_other,
                 imgB_s1_other, _, _, ignore_mask_other, _, _, 
                 vl_mask_change_other, vl_mask_A_other, vl_mask_B_other)) in enumerate(loader):
            t0 = time.time()
            iters = epoch * len(trainloader_u) + i
            imgA_x, imgB_x = imgA_x.cuda(), imgB_x.cuda()
            imgA_s1, imgB_s1 = imgA_s1.cuda(), imgB_s1.cuda()
            mask_x = mask_x.cuda()
            imgA_w, imgB_w = imgA_w.cuda(), imgB_w.cuda()
            ignore_mask = ignore_mask.cuda()
            mix1 = mix1.cuda()
            imgA_w_other, imgB_w_other = imgA_w_other.cuda(), imgB_w_other.cuda()
            imgA_s1_other, imgB_s1_other = imgA_s1_other.cuda(), imgB_s1_other.cuda()
            ignore_mask_other = ignore_mask_other.cuda()

            # for VL pseudo labels
            vl_mask_x_change = vl_mask_x_change.cuda(0)
            vl_mask_x_A, vl_mask_x_B = vl_mask_x_A.cuda(), vl_mask_x_B.cuda()
            vl_mask_A, vl_mask_B = vl_mask_A.cuda(), vl_mask_B.cuda()
            vl_mask_A_other, vl_mask_B_other = vl_mask_A_other.cuda(), vl_mask_B_other.cuda()
            vl_mask_change, vl_mask_change_other = vl_mask_change.cuda(), vl_mask_change_other.cuda()

            # CutMix images
            cutmix_img_(imgA_s1, imgA_s1_other, mix1)
            cutmix_img_(imgB_s1, imgB_s1_other, mix1)

            # Generate pseudo labels
            with torch.no_grad():
                model.eval()

                pred_w_other = model(imgA_w_other, imgB_w_other).detach()
                conf_w_other, mask_w_other = pred_w_other.softmax(dim=1).max(dim=1)

            # Generate predictions
            model.train()

            preds, preds_for_vl, preds_segA, preds_segB, preds_dist = model(torch.cat((imgA_x, imgA_w)), torch.cat((imgB_x, imgB_w)), need_seg_aux=True, need_contrast=True)
            pred_x, pred_w = preds.chunk(2)
            pred_x_for_vl, pred_w_for_vl = preds_for_vl.chunk(2)
            pred_x_segA, pred_w_segA = preds_segA.chunk(2)
            pred_x_segB, pred_w_segB = preds_segB.chunk(2)
            pred_x_dist, pred_w_dist = preds_dist.chunk(2)

            pred_s, pred_s_for_vl, pred_s_segA, pred_s_segB, pred_s_dist = model(imgA_s1, imgB_s1, need_seg_aux=True, need_contrast=True)

            pred_w = pred_w.detach()
            conf_w, mask_w = pred_w.softmax(dim=1).max(dim=1)

            # generate label via distance
            # mask_w_dist = (pred_w_dist.detach() > 1).long()

            # CutMix labels
            mask_w_mixed1 = cutmix_mask(mask_w, mask_w_other, mix1)
            conf_w_mixed1 = cutmix_mask(conf_w, conf_w_other, mix1)
            ignore_mask_mixed1 = cutmix_mask(ignore_mask, ignore_mask_other, mix1)

            if vl_consistency_lambda != 0:
                vl_mask_change_mixed1= cutmix_mask(vl_mask_change, vl_mask_change_other, mix1)
                vl_mask_A_mixed1 = cutmix_mask(vl_mask_A, vl_mask_A_other, mix1)
                vl_mask_B_mixed1 = cutmix_mask(vl_mask_B, vl_mask_B_other, mix1)

            # Supervised Loss
            if criterion_l is not None:
                loss_x = criterion_l(pred_x, mask_x)
                loss_x_dist = criterion_dist(pred_x_dist, mask_x, reduction='mean')
            else:
                losses = model.module.decode_head.loss_decode({'pred_masks': pred_x}, mask_x)
                loss_x, log_vars_x = model.module._parse_losses(losses)
            if vl_consistency_lambda != 0:
                loss_x_A = criterion_l(pred_x_segA, vl_mask_x_A)
                loss_x_B = criterion_l(pred_x_segB, vl_mask_x_B)
                loss_x_vl = criterion_l(pred_x_for_vl, vl_mask_x_change)

            # FixMatch Loss
            if criterion_u is not None:
                loss_s1 = criterion_u(pred_s, mask_w_mixed1)
                loss_s1 = confidence_weighted_loss(loss_s1, conf_w_mixed1, ignore_mask_mixed1, cfg)

                loss_s1_dist = criterion_dist(pred_s_dist, mask_w_mixed1, reduction='none')
                loss_s1_dist = confidence_weighted_loss(loss_s1_dist, conf_w_mixed1, ignore_mask_mixed1, cfg)
            else:
                loss_s1, _ = model.module._parse_losses(
                    model.module.decode_head.loss_decode({'pred_masks': pred_s}, mask_w_mixed1))
                conf_ratio = ((conf_w_mixed1 >= cfg['conf_thresh']) & (ignore_mask_mixed1 != 255)).sum().item() / \
                    (ignore_mask_mixed1 != 255).sum().item()
                loss_s1 *= conf_ratio
            if vl_consistency_lambda != 0:
                loss_vl_s1 = compute_vl_loss(pred_s_for_vl, vl_mask_change_mixed1, ignore_mask_mixed1)
                loss_vl_w = compute_vl_loss(pred_w_for_vl, vl_mask_change, ignore_mask)
                # seg
                loss_vl_s1_A = compute_vl_loss(pred_s_segA, vl_mask_A_mixed1, ignore_mask_mixed1)
                loss_vl_s1_B = compute_vl_loss(pred_s_segB, vl_mask_B_mixed1, ignore_mask_mixed1)
                loss_vl_w_A = compute_vl_loss(pred_w_segA, vl_mask_A, ignore_mask)
                loss_vl_w_B = compute_vl_loss(pred_w_segB, vl_mask_B, ignore_mask)

            if vl_consistency_lambda != 0:
                if isinstance(vl_consistency_lambda, list) or isinstance(vl_consistency_lambda, tuple):
                    assert len(vl_consistency_lambda) == 2
                    prog = iters / total_iters
                    current_vl_lambda = vl_consistency_lambda[0] * (1 - prog) + vl_consistency_lambda[1] * prog
                else:
                    current_vl_lambda = vl_consistency_lambda

            loss = (loss_x + loss_s1) / 2.0
            # Contrastive loss
            loss = loss + (loss_x_dist + loss_s1_dist) / 2.0 * contrastive_loss_weight

            if vl_consistency_lambda != 0:
                loss = loss + loss_x_vl * current_vl_lambda / 6.0
                loss = loss + loss_vl_s1 * current_vl_lambda / 6.0
                loss = loss + loss_vl_w * current_vl_lambda / 6.0

                loss = loss + (loss_x_A + loss_x_B) * 0.5 * current_vl_lambda / 6.0
                loss = loss + (loss_vl_s1_A + loss_vl_s1_B) * 0.5 * current_vl_lambda / 6.0
                loss = loss + (loss_vl_w_A + loss_vl_w_B) * 0.5 * current_vl_lambda / 6.0

            torch.distributed.barrier()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if 'optimizer' not in cfg:
                if iters < cfg['warmup_iters']:
                    k = (1 - iters / cfg['warmup_iters']) * (1 - cfg['warmup_ratio'])
                    lr = cfg['lr'] * (1 - k)
                else:
                    lr = cfg['lr'] * (1 - iters / scheduler_max_iters) ** 0.9
                optimizer.param_groups[0]["lr"] = lr
                optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            else:
                if iters < cfg['warmup_iters']:
                    k = (1 - iters / cfg['warmup_iters']) * (1 - cfg['warmup_ratio'])
                    for group in optimizer.param_groups:
                        group['lr'] = group['initial_lr'] * (1 - k)
                else:
                    for group in optimizer.param_groups:
                        group['lr'] = group['initial_lr'] * (1 - iters / scheduler_max_iters) ** 0.9

            # Logging
            log_avg.update({
                'train/iter_time': time.time() - t0,
                'train/loss_all': loss,
                'train/loss_x': loss_x,
                'train/loss_s1': loss_s1,
            })
            if vl_consistency_lambda != 0:
                log_avg.update({
                    'train/loss_vl_s1': loss_vl_s1,
                    'train/loss_x_seg': loss_x_A + loss_x_B,
                    'train/loss_vl_s1_seg': loss_vl_s1_A + loss_vl_s1_B,
                    'train/loss_vl_w_seg': loss_vl_w_A + loss_vl_w_B,
                })

            if i % 100 == 0 and rank == 0:
                logger.info(f'Iters: {i} ' + str(log_avg))
                for k, v in log_avg.avgs.items():
                    writer.add_scalar(k, v, iters)

                log_avg.reset()

            if iters % len(trainloader_u) == 0 and rank == 0:
                print('Save debug images at iteration', iters)
                out_dir = os.path.join(save_path, 'debug')
                os.makedirs(out_dir, exist_ok=True)
                for b_i in range(imgA_x.shape[0]):
                    rows, cols = 4, 4
                    plot_dicts = [
                        dict(title='ImageA L', data=imgA_x[b_i], type='image'),
                        dict(title='ImageA S1', data=imgA_s1[b_i], type='image'),
                        dict(title='ImageA, VL_mask', data=vl_mask_A_mixed1[b_i], type='label', palette=palette),
                        dict(title='ImageA FP', data=imgA_w[b_i], type='image'),
                        dict(title='ImageB L', data=imgB_x[b_i], type='image'),
                        dict(title='ImageB S1', data=imgB_s1[b_i], type='image'),
                        dict(title='ImageB, VL_mask', data=vl_mask_B_mixed1[b_i], type='label', palette=palette),
                        dict(title='ImageB FP', data=imgB_w[b_i], type='image'),
                        dict(title='Pred L', data=pred_x[b_i], type='prediction', palette=palette),
                        dict(title='Pred S1', data=pred_s[b_i], type='prediction', palette=palette),
                        dict(title='ImageA, W_mask_A', data=pred_s_segA[b_i], type='prediction', palette=palette),
                        dict(title='ImageB, W_mask_B', data=pred_s_segB[b_i], type='prediction', palette=palette),
                        dict(title='GT L', data=mask_x[b_i], type='label', palette=palette),
                        dict(title='PL S1', data=mask_w_mixed1[b_i], type='label', palette=palette),
                        None,
                        dict(title='PL FP', data=mask_w[b_i], type='label', palette=palette),
                    ]
                    if vl_consistency_lambda != 0:
                        plot_dicts.extend([
                            None,
                            dict(title='VL S1', data=vl_mask_change_mixed1[b_i], type='label', palette=palette),
                            None,
                            dict(title='VL FP', data=vl_mask_change[b_i], type='label', palette=palette),
                        ])
                        rows += 1
                    fig, axs = plt.subplots(
                        rows, cols, figsize=(2 * cols, 2 * rows), squeeze=False, 
                        gridspec_kw={'hspace': 0.1, 'wspace': 0, 'top': 0.95, 'bottom': 0, 'right': 1, 'left': 0})
                    for ax, plot_dict in zip(axs.flat, plot_dicts):
                        if plot_dict is not None:
                            plot_data(ax, **plot_dict)
                    plt.savefig(os.path.join(out_dir, f'{(iters):07d}_{rank}-{b_i}.png'))
                    plt.close()

        if epoch % cfg.get('eval_every_n_epochs', 1) == 0 or epoch == cfg['epochs'] - 1:
            eval_mode = cfg['eval_mode']
            mIoU, iou_class, overall_acc, f1_class, precision_class, recall_class = evaluate(model, valloader, eval_mode, cfg, return_cd_metric=True)

            if rank == 0:
                logger.info(run_name)
                logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}'.format(eval_mode, mIoU))
                logger.info('***** Evaluation ***** >>>> IoU (Unchanged/Changed): {:.2f}/{:.2f}'.format(iou_class[0], iou_class[1]))
                logger.info('***** Evaluation ***** >>>> F1 (Unchanged/Changed): {:.2f}/{:.2f}'.format(f1_class[0], f1_class[1]))
                logger.info('***** Evaluation ***** >>>> Precision (Unchanged/Changed): {:.2f}/{:.2f}'.format(precision_class[0], precision_class[1]))
                logger.info('***** Evaluation ***** >>>> Recall (Unchanged/Changed): {:.2f}/{:.2f}'.format(recall_class[0], recall_class[1]))
                logger.info('***** Evaluation ***** >>>> Overall Accuracy: {:.2f}\n'.format(overall_acc))
                
                writer.add_scalar('eval/mIoU', mIoU, epoch)
                for i, iou in enumerate(iou_class):
                    writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

            is_best = iou_class[1] > previous_best_iou
            previous_best_iou = max(iou_class[1], previous_best_iou)
            if is_best:
                previous_best_acc = overall_acc
                
            if rank == 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                }
                if is_best:
                    torch.save(checkpoint, os.path.join(save_path, 'best.pth'))
