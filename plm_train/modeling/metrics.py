import torch

def perplexity(probs):
    assert ((probs.sum(-1) - 1) < 1e-5).all()
    entropy = -(probs * torch.log2(probs + 1e-8)).sum(-1)
    return 2 ** entropy


def compute_standard_metrics(logits_dropped, labels_dropped):
    acc1 = (logits_dropped.argmax(dim=-1) == labels_dropped).float().mean().item()
    return {
        'acc1': acc1,
    }
