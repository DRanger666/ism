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
        ## Two main heads (lm_head for amino acid prediction, contact_head for structure prediction)
        self.backbone.contact_head.requires_grad_(False)

    def get_alphabet(self):
        return 'esm', self.alphabet

    def forward(self, x, batch, remove_start_end=True):
        '''
        ESM2 Layer Structure:
        Layer 0:  Input Embeddings
        Layer 1:  Transformer Block 1
        Layer 2:  Transformer Block 2
        ...
        Layer 32: Transformer Block 32
        Layer 33: Final Layer (for 33-layer model)
        '''
        # repr_layers: Selects which transformer layers to extract embeddings from; (ISM uses final layer only)
        # The final layer typically provides the best performance for downstream tasks
        x = self.backbone(x, repr_layers=[self.num_layers])['representations'][self.num_layers]
        # Handle MSA inputs (batch, msa_depth, seq_len, hidden)
        if len(x.shape) == 4:  # remove MSAs
            # Take first sequence from MSA
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
