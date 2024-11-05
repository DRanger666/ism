import torch
import torch.nn as nn

import esm

from modeling.utils import FFNLayer
from data import one_letters

class ESMBackbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone, self.alphabet = getattr(esm.pretrained, args.backbone)()
        self.num_layers = len(self.backbone.layers)
        self.hdim = self.backbone.lm_head.dense.weight.shape[1]

        if args.freeze_at > 0:
            self.backbone.embed_tokens.requires_grad_(False)
            for i, layer in enumerate(self.backbone.layers):
                if i < args.freeze_at:
                    layer.requires_grad_(False)

        ## prepare feature extractor
        self.ln = nn.LayerNorm(self.hdim)
        self.bb_adapter = FFNLayer(self.hdim, self.hdim)

        ## avoid distributed issues
        self.backbone.lm_head.requires_grad_(False)
        self.backbone.contact_head.requires_grad_(False)

    def get_aa_embed_dim(self):
        return self.aa_embed.shape[1]

    def get_aa_embed(self):
        return self.ln_head(self.aa_embed[self.esm_to_our_aatype])

    def get_alphabet(self):
        return 'esm', self.alphabet

    def forward(self, x, batch):
        x = self.backbone(x, repr_layers=[self.num_layers])['representations'][self.num_layers]
        x = self.bb_adapter(self.ln(x))
        x = x[:, 1:-1]  # remove SOS and EOS tokens
        ret = {'bb_feat': x}
        return ret


def create_backbone(args):
    if 'esm' in args.backbone.lower():
        backbone = ESMBackbone(args)
    else:
        assert False, f'unknown backbone type {args.backbone}'
    return backbone
