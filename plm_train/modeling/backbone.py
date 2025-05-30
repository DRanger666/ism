import torch
import torch.nn as nn

import esm


class ESMBackbone(nn.Module):
    def __init__(self, args):
        """Initialize ESM2 backbone with optional layer freezing"""
        
        super().__init__()
        self.args = args
        # args.backbone is a string like "esm2_t33_650M_UR50D"
        # getattr(esm.pretrained, "esm2_t33_650M_UR50D") 
        # - retrieves the function esm.pretrained.esm2_t33_650M_UR50D
        self.backbone, self.alphabet = getattr(esm.pretrained, args.backbone)()

        self.num_layers = len(self.backbone.layers)
        # ESM2 has TWO main heads:
        # - Language Modeling Head (lm_head) for amino acid prediction 
        # - Contact Prediction Head (contact_head) for structure prediction
        # self.backbone.lm_head.dense.weight.shape = [embed_dim, embed_dim]
        # - For ESM2-650M: [1280, 1280]
        # - shape[1] gets the second dimension = 1280
        # - So, self.hdim = 1280 (the hidden dimension)
        self.hdim = self.backbone.lm_head.dense.weight.shape[1]

        if args.freeze_at > 0:
            self.backbone.embed_tokens.requires_grad_(False)
            for i, layer in enumerate(self.backbone.layers):
                if i < args.freeze_at:
                    layer.requires_grad_(False)

        # avoid distributed issues
        self.backbone.contact_head.requires_grad_(False)

    def get_alphabet(self):
        return 'esm', self.alphabet

    def forward(self, x, batch, remove_start_end=True):
        """
        Forward pass through ESM2 backbone.

        Args:
            x: Input tensor
            batch: Batch information
            remove_start_end: Whether to remove SOS/EOS tokens

        Returns:
            dict: Dictionary with 'bb_feat' key containing embeddings

        Note:
            repr_layers parameter controls which transformer layers to extract:
            - [33]: Extract only final layer (used here)
            - [0, 16, 32, 33]: Extract multiple specific layers
            - list(range(34)): Extract all layers (0-33)
        """
        
        # ESM2 Layer Structure:
        # Layer 0:  Input Embeddings
        # Layer 1:  Transformer Block 1
        # Layer 2:  Transformer Block 2
        # ...
        # Layer 32: Transformer Block 32
        # Layer 33: Final Layer (for 33-layer model)
        
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
    """Create backbone model based on configuration"""
    
    if 'esm' in args.backbone.lower():
        backbone = ESMBackbone(args)
    else:
        assert False, f'unknown backbone type {args.backbone}'
    return backbone
