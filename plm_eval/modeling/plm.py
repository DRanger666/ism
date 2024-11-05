import torch
import torch.nn as nn
from modeling.backbone import create_backbone

VOCAB_SIZE = 33

class PLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = create_backbone(args)

        if 'secondary' in args.data_path:
            K = 3
        else:
            K = 1

        self.classifier = nn.Linear(self.backbone.hdim, K)

    def forward(self, x, batch):
        pred = {}
        pred.update(self.backbone(x, batch))
        pred['logit'] = self.classifier(pred['bb_feat'])
        return pred
