import torch.nn as nn

from modeling.backbone import create_backbone
VOCAB_SIZE = 33

class PLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = create_backbone(args)

        self.head = self.backbone.backbone.lm_head

        self.use_evo_aae_heads = 'allmergedce' in args.loss_func
        if self.use_evo_aae_heads:
            self.evo_head = nn.Linear(self.backbone.hdim, 512+1)  # padding is 0
            self.aae_head = nn.Linear(self.backbone.hdim, 64+1)

    def forward(self, x, batch):
        pred = {}
        pred.update(self.backbone(x, batch))
        feats = pred['bb_feat']
        logits = self.head(feats)
        pred['logits'] = logits
        pred['evo_logits'] = self.evo_head(feats) if self.use_evo_aae_heads else None
        pred['aae_logits'] = self.aae_head(feats) if self.use_evo_aae_heads else None
        return pred
