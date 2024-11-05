import math
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import misc
import wandb

import sklearn.metrics as sk_metrics

def train_one_epoch(model: torch.nn.Module,
                    dl, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    args=None):
    ## prepare training
    model.train(True)
    optimizer.zero_grad()

    ## prepare logging
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 10

    for batch_idx, batch in enumerate(metric_logger.log_every(dl, print_freq, header)):
        misc.adjust_learning_rate(optimizer, batch_idx / len(dl) + epoch, args)

        ## move inputs/outputs to cuda
        x = batch['tokens'].to(device, non_blocking=True)
        pad_mask = batch['pad_mask'].to(device)
        if 'secondary' in args.data_path:
            label = batch['ss_label'].to(device)
            ss_mask = batch['valid_ss_mask'].to(device)
        else:
            bind_label = batch['bind_label'].to(device)

        ## forward
        pred = model(x, batch)
        logit = pred['logit']
        if 'secondary' in args.data_path:
            loss = F.cross_entropy(logit[~pad_mask & ss_mask], label[~pad_mask & ss_mask])
            acc = (logit.argmax(-1) == label)[~pad_mask & ss_mask].float().mean()
        else:
            logit = logit.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logit[~pad_mask], bind_label[~pad_mask])
            acc = ((logit > 0) == bind_label)[~pad_mask].float().mean()

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        ## backward
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        optimizer.zero_grad()

        ## logging
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss.item())
        metric_logger.update(acc=acc.item())
        if not args.disable_wandb and misc.is_main_process():
            wandb.log({
                'train_loss': loss.item(),
                'lr': lr,
                'train_acc': acc.item(),
            })

    ## gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, dl, device, args):
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    all_preds = {}
    loss_list = []
    for batch in metric_logger.log_every(dl, 10, header):
        ## move inputs/outputs to cuda
        x = batch['tokens'].to(device, non_blocking=True)
        pad_mask = batch['pad_mask'].to(device)
        if 'secondary' in args.data_path:
            label = batch['ss_label'].to(device)
            ss_mask = batch['valid_ss_mask'].to(device)
        else:
            bind_label = batch['bind_label'].to(device)

        ## forward
        pred = model(x, batch)
        logit = pred['logit']
        if 'secondary' in args.data_path:
            ss_prob = F.softmax(logit, dim=-1)
            loss = F.cross_entropy(logit[~pad_mask & ss_mask], label[~pad_mask & ss_mask])
        else:
            logit = logit.squeeze(-1)
            bind_prob = logit.sigmoid()
            loss = F.binary_cross_entropy_with_logits(logit[~pad_mask], bind_label[~pad_mask])
        loss_list.append(loss.item())

        ## format for eval
        for b in range(len(x)):

            if 'secondary' in args.data_path:
                pdb_id = batch['pdb_id'][b]
                ss_label_b = label[b][~pad_mask[b] & ss_mask[b]].cpu().numpy()
                ss_prob_b = ss_prob[b][~pad_mask[b] & ss_mask[b]].cpu().numpy()
                all_preds[pdb_id] = {
                    'ss_label': ss_label_b,
                    'ss_prob': ss_prob_b,
                }

            else:
                uniprot = batch['uniprot'][b]
                bind_label_b = bind_label[b][~pad_mask[b]].cpu().numpy()
                bind_prob_b = bind_prob[b][~pad_mask[b]].cpu().numpy()
                all_preds[uniprot] = {
                    'bind_label': bind_label_b,
                    'bind_prob': bind_prob_b,
                }

    if args.dist_eval:
        print('Start gathering predictions')
        torch.cuda.empty_cache()
        all_preds = misc.gather_dict_keys_on_main(all_preds)
        print(f'Finished gathering predictions')
        if not misc.is_main_process():
            return {}

    ret = {}
    ds_name = dl.dataset.name
    if 'secondary' in args.data_path:
        all_ss_labels = np.concatenate([v['ss_label'] for v in all_preds.values()])
        all_ss_probs = np.concatenate([v['ss_prob'] for v in all_preds.values()])
        all_ss_preds = all_ss_probs.argmax(-1)
        macro_acc = (all_ss_preds == all_ss_labels).mean()
        pdb_acc = [(v['ss_prob'].argmax(-1) == v['ss_label']).mean() for v in all_preds.values()]
        micro_acc = np.mean([acc for acc in pdb_acc if acc == acc])
        metrics = {
            'macro_acc': macro_acc,
            'micro_acc': micro_acc,
            'loss': sum(loss_list) / len(loss_list),
        }
        COPYPAST_NAMES_SS = ['macro_acc', 'micro_acc']
        ret['copypasta'] = ','.join([f'{metrics[name]:.02f}' for name in COPYPAST_NAMES_SS])
    else:
        all_bind_labels = np.concatenate([v['bind_label'] for v in all_preds.values()])
        all_bind_probs = np.concatenate([v['bind_prob'] for v in all_preds.values()])
        metrics = {
            'f1':  sk_metrics.f1_score(all_bind_labels, all_bind_probs > 0.5),
            'mcc': sk_metrics.matthews_corrcoef(all_bind_labels, all_bind_probs > 0.5),
            'auc': sk_metrics.roc_auc_score(all_bind_labels, all_bind_probs),
            'loss': sum(loss_list) / len(loss_list),
        }
        COPYPASTE_NAMES_BIND = ['f1', 'mcc', 'auc']
        ret['copypasta'] = ','.join([f'{metrics[name]:.02f}' for name in COPYPASTE_NAMES_BIND])

    metric_logger.update(**metrics)
    ret.update({k: meter.global_avg for k, meter in metric_logger.meters.items()})
    ret = {f'{ds_name}_{k}': v for k, v in ret.items()}
    print(ds_name, ret)
    if not args.disable_wandb and misc.is_main_process():
        wandb.log(ret)
    return ret
