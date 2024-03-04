import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from progress.bar import Bar

from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy
from utils import Logger

logger = logging.getLogger(__name__)
best_acc = 0


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar', epoch_p=1):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    seed = args.seed
    print("Setting fixed seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def ce_loss(logits, targets, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.

    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    assert logits.shape == targets.shape
    log_pred = F.log_softmax(logits, dim=-1)
    nll_loss = torch.sum(-targets * log_pred, dim=1)
    return nll_loss


def main():
    parser = argparse.ArgumentParser(description='PyTorch UniMatch Training')
    parser.add_argument('--gpu-id', default='3', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'],
                        help='dataset name')

    parser.add_argument('--rl', type=float, default=0.2,
                        help='ratio of labeled data')
    parser.add_argument('--rp', type=float, default=0.4,
                        help='ratio of partial-labeled data')
    parser.add_argument('--ru', type=float, default=0.4,
                        help='ratio of unlabeled data')
    parser.add_argument('-partial_type', help='flaverage_class_labelipping strategy', type=str, default='binomial',
                        choices=['binomial', 'pair'])
    parser.add_argument('-partial_rate', help='flipping probability', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='EMA ratio for soft label update')
    parser.add_argument('--feature_dim', type=int, default=128,
                        help='feature_dim for projection head')

    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=500000, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=500, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')

    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")

    args = parser.parse_args()
    global best_acc

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes,
                                            feature_dim=args.feature_dim)
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters()) / 1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, ")

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    train_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, './data')

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    title = 'UniMatch-' + args.dataset
    args.logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
    args.logger.set_names(['Top1 acc', 'Top5 acc', 'Best Top1 acc'])

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(
        f"  Task = {args.dataset}@labeled:{args.rl * 100}% partial-labeled:{args.rp * 100}% unlabeled:{args.ru * 100}%")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size * args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(args, train_dataset, train_loader, test_loader,
          model, optimizer, ema_model, scheduler)


def train(args, train_dataset, train_loader, test_loader,
          model, optimizer, ema_model, scheduler):
    global best_acc
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        train_epoch = 0
        train_loader.sampler.set_epoch(train_epoch)

    train_iter = iter(train_loader)
    mse_loss = nn.MSELoss(reduction='mean')

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        print('current epoch: ', epoch + 1)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_cls = AverageMeter()
        losses_contrac = AverageMeter()

        bar = Bar('Training', max=args.eval_step)

        for batch_idx in range(args.eval_step):
            try:
                (w_img, s_img, s1_img), label, mask, idx = next(train_iter)
            except:
                if args.world_size > 1:
                    train_epoch += 1
                    train_loader.sampler.set_epoch(train_epoch)
                train_iter = iter(train_loader)
                (w_img, s_img, s1_img), label, mask, idx = next(train_iter)

            mask = mask.to(args.device)
            # idx = idx.to(args.device)
            data_time.update(time.time() - end)
            batch_size = w_img.shape[0]

            all_img = torch.cat([w_img, s_img, s1_img], dim=0).to(args.device)
            label = label.to(args.device)
            feat_all = model(all_img)

            # classification-part
            lgt_all = model.f(feat_all)
            lgt_w = lgt_all[:batch_size]
            lgt_s, lgt_s1 = lgt_all[batch_size:].chunk(2)
            del lgt_all

            psd_label = torch.softmax(lgt_w.detach() / args.T, dim=-1)
            psd_label = psd_label * mask
            psd_label = psd_label / (psd_label.sum(1, keepdim=True) + 1e-4)

            loss_cls = (ce_loss(lgt_s, psd_label, reduction='none').mean() + ce_loss(lgt_s1, psd_label,
                                                                                     reduction='none').mean()) / 2
            # update pseudo label for retainment
            train_dataset.update_psdlabel(psd_label.cpu(), idx, args.alpha)

            # # graph-contrastive-part
            # contrac_all = model.g(feat_all)
            # contrac_w = contrac_all[:batch_size]
            # contrac_s, contrac_s1 = contrac_all[batch_size:].chunk(2)
            # del contrac_all

            # contrac_g = torch.cat([contrac_s, contrac_s1], dim=0).to(args.device)
            # sim_g = torch.matmul(contrac_g, contrac_g.T)
            # # calculate contrac-feat norm
            # norms = torch.norm(contrac_g, p=2, dim=1, keepdim=True)
            # norm_factor = torch.matmul(norms, norms.T)
            # sim_g = sim_g / norm_factor

            # contrac_f = torch.cat([label, label], dim=0).to(args.device)
            # sim_f = torch.matmul(contrac_f, contrac_f.T)
            # # idx for different view of same img
            # indices = torch.arange(batch_size)
            # view_indices = indices + batch_size
            # # set similarity 1
            # sim_f[indices, view_indices] = 1.0
            # sim_f[view_indices, indices] = 1.0 
            # sim_f[indices, indices] = 0.0
            # loss_contrac = mse_loss(sim_f, sim_g)

            # loss = loss_cls + loss_contrac
            loss_contrac = loss_cls
            loss = loss_cls
            loss.backward()

            losses.update(loss.item())
            losses_cls.update(loss_cls.item())
            losses_contrac.update(loss_contrac.item())

            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix = '({batch}/{size}) | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                         'Loss: {loss:.4f} | Loss_cls: {loss_cls:.4f} | Loss_contrac: {loss_contrac:.4f} '.format(
                batch=batch_idx + 1,
                size=args.eval_step,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                loss_cls=losses_cls.avg,
                loss_contrac=losses_contrac.avg,
            )
            bar.next()
        bar.finish()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc, test_top5_acc = test(args, test_loader, test_model, epoch)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_cls.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_contrac.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema

            if (epoch + 1) % 10 == 0 or (is_best and epoch > 250):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),
                    'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, is_best, args.out, epoch_p=epoch + 1)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

            args.logger.append([test_acc, test_top5_acc, best_acc])

    if args.local_rank in [-1, 0]:
        args.writer.close()


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            feat = model(inputs)
            outputs = model.f(feat)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    main()
