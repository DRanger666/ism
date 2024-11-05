import torch
import torch.nn as nn

import esm


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

        ## avoid distributed issues
        self.backbone.contact_head.requires_grad_(False)

    def get_alphabet(self):
        return 'esm', self.alphabet

    def forward(self, x, batch, remove_start_end=True):
        x = self.backbone(x, repr_layers=[self.num_layers])['representations'][self.num_layers]
        if len(x.shape) == 4:  # remove MSAs
            x = x[:,0]
        if remove_start_end:
            x = x[:, 1:-1]  # remove SOS and EOS tokens
        ret = {'bb_feat': x}
        return ret

def create_backbone(args):
    if 'esm' in args.backbone.lower():
        backbone = ESMBackbone(args)
    else:
        assert False, f'unknown backbone type {args.backbone}'
    return backbone
