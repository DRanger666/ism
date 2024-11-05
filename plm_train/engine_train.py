import math
import sys

import torch

import misc
import wandb

VOCAB_SIZE = 33

def train_one_epoch(model: torch.nn.Module,
                    dl, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    loss_scaler, criterion,
                    args=None):
    ## prepare training
    model.train(True)
    optimizer.zero_grad()

    ## prepare logging
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 10

    accum_iter = args.accum_iter

    for batch_idx, batch in enumerate(metric_logger.log_every(dl, print_freq, header)):
        fps = batch.pop('fp', [])
        seqs = batch.pop('wt_seq', [])

        if batch_idx % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, batch_idx / len(dl) + epoch, args)

        ## move inputs/outputs to cuda
        for k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)

        ## forward
        with torch.amp.autocast('cuda'):
            pred = model(batch['tokens'], batch)
            loss, metrics = criterion(pred, batch, args)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        ## backward
        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
            clip_grad=5, update_grad=(batch_idx + 1) % accum_iter == 0)
        if (batch_idx + 1) % accum_iter == 0:
            optimizer.zero_grad()

        ## logging
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss.item())
        metric_logger.update(**metrics)
        if not args.disable_wandb and misc.is_main_process():
            wandb.log({
                'train_loss': loss.item(),
                'lr': lr,
                **metrics,
            })

    ## gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, dl, criterion, device, args):
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    for batch_idx, batch in enumerate(metric_logger.log_every(dl, 10, header)):
        fps = batch.pop('fp', [])
        seqs = batch.pop('wt_seq', [])

        ## move inputs/outputs to cuda
        for k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)

        ## forward
        with torch.amp.autocast('cuda'):
            pred = model(batch['tokens'], batch)
            loss, metrics = criterion(pred, batch, args)

        metric_logger.update(loss=loss.item())
        metric_logger.update(**metrics)

    ## gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    ret = {f'val_{k}': meter.global_avg for k, meter in metric_logger.meters.items()}
    if not args.disable_wandb and misc.is_main_process():
        wandb.log(ret)
    return ret
