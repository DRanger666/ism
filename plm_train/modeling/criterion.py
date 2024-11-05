import torch
import torch.nn.functional as F

from . import metrics as metrics_util

def masked_cross_entropy(pred: dict, batch: dict, args) -> tuple[torch.Tensor, dict]:
    logits = pred['logits']
    labels = batch['labels'][:,1:-1]
    drop_mask = batch['drop_mask'][:,1:-1]
    B, L = labels.shape

    labels_dropped = labels[drop_mask]
    logits_dropped = logits[drop_mask]
    loss = F.cross_entropy(logits_dropped, labels_dropped)

    metrics = metrics_util.compute_standard_metrics(logits_dropped, labels_dropped)
    metrics['mr'] = drop_mask.sum() / (labels > 2).sum()
    return loss, metrics

def allmergedce(pred: dict, batch: dict, args, lambda_evo) -> tuple[torch.Tensor, dict]:
    evo_labels = batch['evo_tokens']
    aae_labels = batch['aae_tokens']

    evo_logits = pred['evo_logits']
    aae_logits = pred['aae_logits']

    B, L = evo_labels.shape

    loss_mask = batch['tokens'][:,1:-1] > 2  # 0,1,2 are CLS,PAD,EOS
    aae_mask = loss_mask & (aae_labels != 17 + 1)  # id17 contains unfolded microenvs

    evo_labels_dropped = evo_labels[loss_mask]
    aae_labels_dropped = aae_labels[aae_mask]
    evo_logits_dropped = evo_logits[loss_mask]
    aae_logits_dropped = aae_logits[aae_mask]
    if loss_mask.sum() == 0:
        evo_loss = 0 * evo_logits_dropped.sum()
        aae_loss = 0 * aae_logits_dropped.sum()
    else:
        evo_loss = F.cross_entropy(evo_logits_dropped, evo_labels_dropped)
        if aae_mask.sum() == 0:
            aae_loss = 0 * aae_logits_dropped.sum()
        else:
            aae_loss = F.cross_entropy(aae_logits_dropped, aae_labels_dropped)

    evo_mr = loss_mask.sum().item() / (batch['labels'] > 2).sum().item()
    aae_mr = aae_mask.sum().item() / (batch['labels'] > 2).sum().item()

    evo_metrics = metrics_util.compute_standard_metrics(evo_logits_dropped, evo_labels_dropped)
    aae_metrics = metrics_util.compute_standard_metrics(aae_logits_dropped, aae_labels_dropped)
    metrics = {
        'evo_loss': evo_loss.item(),
        'aae_loss': aae_loss.item(),
        'evo_mr': evo_mr,
        'aae_mr': aae_mr,
        **{f'evo_{k}': v for k, v in evo_metrics.items()},
        **{f'aae_{k}': v for k, v in aae_metrics.items()},
    }
    return evo_loss * lambda_evo + aae_loss, metrics

def allmergedce_ce(pred: dict, batch: dict, args, lambda_evo=1.) -> tuple[torch.Tensor, dict]:
    ce_loss, ce_metrics = masked_cross_entropy(pred, batch, args)
    structce_loss, structce_metrics = allmergedce(pred, batch, args, lambda_evo)
    metrics = {
        'ce_loss': ce_loss.item(),
        'structce_loss': structce_loss.item(),
        **ce_metrics,
        **structce_metrics,
    }
    return ce_loss + structce_loss, metrics
