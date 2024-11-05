import os
import argparse
import datetime
import time
import wandb
import numpy as np
from functools import partial
from pathlib import Path

import torch
import torch.optim as optim

import modeling.criterion as criterion
from modeling.plm import PLM
from engine_train import train_one_epoch, evaluate
from data import protein_collate_fn, create_dataset
import misc

def get_args_parser():
    parser = argparse.ArgumentParser('Protein Language Model')
    parser.add_argument('--seed', default=0, type=int)

    # model params
    parser.add_argument('--backbone', default='esm2_t33_650M_UR50D')
    parser.add_argument('--freeze_at', default=0, type=int, help='freeze backbone up to layer X')

    # data params
    parser.add_argument('--ds_type', type=str, default='sequence_structure_tokens')
    parser.add_argument('--data_path', type=str, default='/path/to/data')
    parser.add_argument('--eval_data_path', type=Path, default=None)
    parser.add_argument('--num_workers', default=72, type=int)
    parser.add_argument('--crop_len', default=512, type=int)

    # train params
    parser.add_argument('--loss_func', type=str, default='masked_cross_entropy')
    parser.add_argument('--mask_ratio', type=float, default=0.15)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-9)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--warmup_epochs', type=int, default=4)
    parser.add_argument('--accum_iter', type=int, default=1)

    # eval params
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--dist_eval', action='store_true')

    # resume params
    parser.add_argument('--finetune', default='', type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--start_epoch', type=int, default=0)

    # logging params
    parser.add_argument('--output_dir', type=Path, default='logs/plm')
    parser.add_argument('--vis_dir', type=Path, default=None, help='only eval')
    parser.add_argument('--eval_period', type=int, default=1)
    parser.add_argument('--save_period', type=int, default=1)
    parser.add_argument('--disable_wandb', action='store_true')
    parser.add_argument('--wandb_group', default=None, type=str)

    # distributed training parameters
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def main(args):
    misc.init_distributed_mode(args)
    if not args.disable_wandb and misc.is_main_process():
        run_name = args.output_dir.name
        wandb.init(project='plm', name=run_name, config=args, dir=args.output_dir, group=args.wandb_group)
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    ## prepare model
    model = PLM(args)
    args.alphabet = model.backbone.get_alphabet()
    n_params = sum(p.numel() for p in model.parameters())
    n_params_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f'Training {n_params_grad} of {n_params} parameters')
    loss_func = getattr(criterion, args.loss_func)

    ## prepare DDP
    model.to(args.device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()

    ## prepare optimizer
    param_groups = misc.param_groups_weight_decay(model, args.weight_decay)
    optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    loss_scaler = misc.NativeScalerWithGradNormCount()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    ## prepare train data
    ds = create_dataset(args)
    test_indices = list(range(0, len(ds), 100))
    train_indices = list(set(range(len(ds))) - set(test_indices))
    ds_train = torch.utils.data.Subset(ds, train_indices)
    ds_test = create_dataset(args, train=False) if args.eval_data_path else torch.utils.data.Subset(ds, test_indices)
    collate_fn = partial(protein_collate_fn, alphabet=args.alphabet, args=args)
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            ds_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(ds_train)
    dl_train = torch.utils.data.DataLoader(ds_train, sampler=sampler_train, batch_size=args.batch_size, collate_fn=collate_fn)
    print(f'Training on {len(ds_train)} samples. Testing on {len(ds_test)} samples')

    if args.distributed and args.dist_eval:
        sampler_test = torch.utils.data.DistributedSampler(ds_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        sampler_test = torch.utils.data.SequentialSampler(ds_test)
    dls_test = [torch.utils.data.DataLoader(ds_test, sampler=sampler_test, batch_size=args.batch_size, collate_fn=collate_fn)]

    if args.eval:
        metrics = {}
        for dl_test in dls_test:
            metrics.update(evaluate(model, dl_test, loss_func, device, args))
        if not args.disable_wandb and misc.is_main_process():
            wandb.finish()
        exit()

    print(f"Start training for {args.epochs} epochs, saving to {args.output_dir}")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            dl_train.sampler.set_epoch(epoch)
        train_one_epoch(model, dl_train, optimizer, device, epoch, loss_scaler, loss_func, args)
        if epoch % args.save_period == args.save_period - 1:
            ckpt_path = misc.save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler=None)
            print(f'Saved checkpoint to {ckpt_path}')
        if epoch % args.eval_period == args.eval_period - 1:
            for dl_test in dls_test:
                evaluate(model, dl_test, loss_func, device, args)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    metrics = {}
    for dl_test in dls_test:
        metrics.update(evaluate(model, dl_test, loss_func, device, args))

    if not args.disable_wandb and misc.is_main_process():
        wandb.finish()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
