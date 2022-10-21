# --------------------------------------------------------
# Copyright (C) 2022 NVIDIA Corporation. All rights reserved.
# 
# Official PyTorch implementation of NeurIPS2022 paper
# Structural Pruning via Latency-Saliency Knapsack
# Maying Shen, Hongxu Yin, Pavlo Molchanov, Lei Mao, Jianna Liu and Jose M. Alvarez
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# --------------------------------------------------------

import os
import time

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to run this example."
    )
import torch
from torch.autograd import Variable

from utils.utils import accuracy, AverageMeter, calc_ips, get_logger, reduce_tensor


global_iter = 0
logger = get_logger("train")

def train_loop(
    exp_cfg,
    model,
    criterion,
    optimizer,
    lr_scheduler,
    train_loader,
    val_loader,
    batch_size_multiplier=1,
    train_writer=None,
    pruner=None,
):
    best_prec1, best_prec5 = 0, 0
    global global_iter
    is_main = (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)
    
    total_iter = exp_cfg.epochs * len(train_loader.dataloader)
    for epoch_id in range(exp_cfg.epochs):
        # Train one epoch
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        data_time = AverageMeter()
        compute_time = AverageMeter()
        ips = AverageMeter()

        model.train()

        step = get_train_step(
            model, criterion, optimizer, exp_cfg.fp16, use_amp=exp_cfg.amp, pruner=pruner
        )

        end = time.time()
        optimizer.zero_grad()

        for i, (input, target) in enumerate(train_loader):
            model.train()
            lr_scheduler(optimizer, epoch_id)
            data_load_time = time.time() - end

            optimizer_step = ((i + 1) % batch_size_multiplier) == 0
            loss, prec1, prec5 = step(input, target, global_iter, optimizer_step=optimizer_step)

            it_time = time.time() - end
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item())
            top5.update(prec5.item())
            data_time.update(data_load_time/input.size(0)*1000)
            compute_time.update((it_time-data_load_time)/input.size(0)*1000)
            ips.update(calc_ips(input.size(0), it_time - data_load_time))

            if (i+1) % 100 == 0:
                if is_main:
                    train_writer.add_scalar("train/iter_loss", losses.val, global_iter)
                    train_writer.add_scalar("train/iter_top1", top1.val, global_iter)
                logger.info(
                    "GlobalIter {}/{}: Epoch {}, Iter {}, loss {}, top1 {}, top5 {}, "
                    "data_time(ms/image) {}, compute_time(ms/image) {}, ips {}".format(
                    global_iter, total_iter, epoch_id, i+1, losses.avg, top1.avg, top5.avg,
                    data_time.avg, compute_time.avg, ips.avg)
                )

            if pruner is not None:
                pruned_num = pruner.prune_step(global_iter)
            global_iter += 1

            end = time.time()

        torch.cuda.synchronize()

        logger.info(
            "Epoch {} Train ===== loss {}, top1 {}, top5 {}, "
            "data_time {}, compute_time {}, ips {}".format(
            epoch_id, losses.avg, top1.avg, top5.avg, data_time.avg,
            compute_time.avg, ips.avg)
        )

        prec1, prec5 = validate(val_loader, model, criterion)

        if is_main:
            train_writer.add_scalar("train/epoch_loss", losses.avg, epoch_id)
            train_writer.add_scalar("train/LR", optimizer.param_groups[0]['lr'], epoch_id)
            train_writer.add_scalar("val/top1", prec1, epoch_id)

        is_best = prec1 > best_prec1
        if is_best:
            best_prec1 = prec1
            best_prec5 = prec5
        if is_main:
            if (epoch_id + 1) % exp_cfg.ckpt_freq == 0 or epoch_id == exp_cfg.epochs - 1:
                save_path = os.path.join(
                    os.environ["OUTPUT_PATH"], "epoch_{}.pth".format(epoch_id)
                )
                torch.save(
                    {
                        "state_dict": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch_id,
                    },
                    save_path
                )
                logger.info("Saved model to {}.".format(save_path))
            if is_best:
                save_path = os.path.join(os.environ["OUTPUT_PATH"], "best_net.pth")
                torch.save(model.state_dict(), save_path)

    return best_prec1, best_prec5


def get_train_step(model, criterion, optimizer, fp16, use_amp, pruner=None):
    def _step(input, target, global_step, optimizer_step=True):
        optimizer.zero_grad()

        input_var = Variable(input)
        target_var = Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        if torch.isnan(loss):
            logger.warning("NAN in loss, skipping the batch and update")
            reduced_loss, prec1, prec5 = torch.zeros(1), torch.zeros(1), torch.zeros(1)
            return reduced_loss, prec1, prec5

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if torch.distributed.is_initialized():
            reduced_loss = reduce_tensor(loss.data)
        else:
            reduced_loss = loss.data

        if fp16:
            optimizer.backward(loss)
        elif use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        has_nan = False
        for w in model.parameters():
            if torch.isnan(w.grad.data.mean()):
                has_nan = True
                break
        if has_nan:
            logger.warning("NAN in grad, skipping the batch and update")
            reduced_loss, prec1, prec5 = torch.zeros(1), torch.zeros(1), torch.zeros(1)
            return reduced_loss, prec1, prec5

        if pruner is not None:
            pruner.update_metric(global_step)

        if optimizer_step:
            optimizer.step()

        torch.cuda.synchronize()

        return reduced_loss, prec1, prec5

    return _step


def get_val_step(model, criterion):
    def _step(input, target):
        input_var = Variable(input)
        target_var = Variable(target)

        with torch.no_grad():
            output = model(input_var)
            loss = criterion(output, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if torch.distributed.is_initialized():
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        torch.cuda.synchronize()

        return reduced_loss, prec1, prec5

    return _step


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    data_time = AverageMeter()
    compute_time = AverageMeter()
    ips = AverageMeter()

    model.eval()
    step = get_val_step(model, criterion)

    end = time.time()

    for i, (input, target) in enumerate(val_loader):
        data_load_time = time.time() - end

        loss, prec1, prec5 = step(input, target)

        it_time = time.time() - end

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item())
        top5.update(prec5.item())
        data_time.update(data_load_time/input.size(0)*1000)
        compute_time.update((it_time - data_load_time)/input.size(0)*1000)
        ips.update(calc_ips(input.size(0), it_time - data_load_time))

        end = time.time()

    logger.info(
        "Validate ===== loss {}, top1 {}, top5 {}, "
        "data_time(ms/image) {}, compute_time(ms/image) {}, ips {}".format(
        losses.avg, top1.avg, top5.avg, data_time.avg, compute_time.avg, ips.avg)
    )

    return top1.avg, top5.avg
