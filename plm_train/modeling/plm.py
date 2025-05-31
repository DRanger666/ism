import torch.nn as nn

from modeling.backbone import create_backbone

# Protein vocabulary size: 20 amino acids + 13 special tokens (CLS, PAD, EOS, MASK, etc.)
# This matches ESM2's original vocabulary and is NOT expanded in ISM
VOCAB_SIZE = 33


class PLM(nn.Module):
    """
    PLM (Protein Language Model) - The core ISM model implementation.
    
    This class implements the ISM architecture which performs "structure-tuning" on ESM2.
    ISM enhances the pre-trained ESM2 model with structural prediction capabilities through
    a multi-task learning approach that combines:
    
    1. Original sequence token prediction (masked language modeling)
    2. Evolutionary environment prediction (512 classes)  
    3. Atomic environment prediction (64 classes)
    
    The key insight is that ISM does NOT expand the input vocabulary - it keeps the same
    33-token ESM2 vocabulary but adds multiple output prediction heads that learn to
    predict structural tokens from the same sequence representations.
    
    This implements a "structural autoencoder" where:
    - Encoder: Pre-computed (3D structure → discrete structural tokens during preprocessing)
    - Decoder: Learned (sequence representations → structural token predictions)
    
    Architecture Overview:
    ┌─────────────────┐
    │  Input Tokens   │ ← Same 33-token ESM2 vocabulary
    │     (33 vocab)  │
    └─────────┬───────┘
              │
    ┌─────────▼───────┐
    │  ESM2 Backbone  │ ← Structure-tuned transformer (650M params)
    │  (Transformer)  │
    └─────────┬───────┘
              │
              ▼ bb_feat (contextual representations)
    ┌─────────┬───────┬─────────┐
    │ LM Head │Evo Head│AAE Head │ ← Multiple prediction heads
    │(33 cls) │(512cls)│(64 cls) │
    └─────────┴───────┴─────────┘
    """
    
    def __init__(self, args):
        """
        Initialize the PLM model with backbone and prediction heads.
        
        Args:
            args: Configuration object containing:
                - backbone: ESM model variant (e.g., 'esm2_t33_650M_UR50D')
                - loss_func: Loss function type (determines which heads to use)
                - freeze_at: Number of layers to freeze during training
        """
        super().__init__()
        self.args = args
        
        # ===== BACKBONE INITIALIZATION =====
        # Create the ESM2 backbone wrapper that handles:
        # - Loading pre-trained ESM2 weights
        # - Optional layer freezing for fine-tuning
        # - Extracting contextual representations from final layer
        self.backbone = create_backbone(args)
        
        # ===== ORIGINAL ESM2 LANGUAGE MODELING HEAD =====
        # Reuse ESM2's original language modeling head for sequence token prediction
        # This head predicts amino acid tokens (33 classes) from contextual features
        # 
        # Architecture: Linear(hidden_dim, 33) with tied weights to input embeddings
        # Purpose: Maintains original ESM2 masked language modeling capability
        self.head = self.backbone.backbone.lm_head
        
        # ===== STRUCTURAL PREDICTION HEADS SETUP =====
        # Determine if we should add structural autoencoder heads
        # This is controlled by the loss function - 'allmergedce' indicates structure-tuning
        self.use_evo_aae_heads = 'allmergedce' in args.loss_func
        
        if self.use_evo_aae_heads:
            # EVOLUTIONARY ENVIRONMENT HEAD
            # Predicts evolutionary conservation patterns and co-evolution signals
            # Input: Contextual features from ESM2 backbone (hidden_dim)
            # Output: 512 evolutionary environment classes + 1 padding class = 513 total
            # 
            # The 512 classes represent discretized evolutionary environments that capture:
            # - Conservation scores across homologous sequences  
            # - Co-evolution patterns with other residues
            # - Phylogenetic depth and evolutionary pressure
            # - Functional constraint classifications
            self.evo_head = nn.Linear(self.backbone.hdim, 512+1)  # +1 for padding token (class 0)
            
            # ATOMIC ENVIRONMENT HEAD  
            # Predicts local atomic/geometric environment around each residue
            # Input: Contextual features from ESM2 backbone (hidden_dim)
            # Output: 64 atomic environment classes + 1 padding class = 65 total
            #
            # The 64 classes represent discretized atomic environments that capture:
            # - Local atomic density and packing
            # - Secondary structure context (alpha helix, beta sheet, loops)
            # - Hydrophobic/hydrophilic environment classification
            # - Coordination patterns and geometric constraints
            self.aae_head = nn.Linear(self.backbone.hdim, 64+1)   # +1 for padding token (class 0)

    def forward(self, x, batch):
        """
        Forward pass through the PLM model.
        
        This implements the core ISM prediction pipeline:
        1. Process sequence tokens through ESM2 backbone 
        2. Extract contextual representations
        3. Make predictions using multiple heads (sequence + structural)
        
        Args:
            x: Input token IDs (batch_size, seq_len) - ESM2 tokenized sequences
            batch: Additional batch data containing structural token targets
            
        Returns:
            pred: Dictionary containing all model predictions:
                - 'bb_feat': Backbone contextual features (batch_size, seq_len, hidden_dim)
                - 'logits': Sequence token predictions (batch_size, seq_len, 33)
                - 'evo_logits': Evolutionary environment predictions (batch_size, seq_len, 513) or None
                - 'aae_logits': Atomic environment predictions (batch_size, seq_len, 65) or None
        """
        # Initialize prediction dictionary to collect all outputs
        pred = {}
        
        # ===== BACKBONE FORWARD PASS =====
        # Process input tokens through the ESM2 backbone transformer
        # This extracts contextual representations that understand protein sequences
        # 
        # The backbone returns:
        # - 'bb_feat': Contextual features from final transformer layer
        #              Shape: (batch_size, seq_len-2, hidden_dim) 
        #              Note: seq_len-2 because CLS/EOS tokens are removed
        pred.update(self.backbone(x, batch))
        
        # Extract the contextual features that will be input to all prediction heads
        # These features contain rich protein sequence understanding from ESM2
        feats = pred['bb_feat']  # Shape: (batch_size, seq_len-2, hidden_dim)
        
        # ===== SEQUENCE TOKEN PREDICTION =====
        # Use original ESM2 language modeling head to predict amino acid tokens
        # This maintains the original masked language modeling capability
        # 
        # Input: Contextual features (batch_size, seq_len-2, hidden_dim)
        # Output: Logits over amino acid vocabulary (batch_size, seq_len-2, 33)
        logits = self.head(feats)
        pred['logits'] = logits
        
        # ===== STRUCTURAL TOKEN PREDICTIONS =====
        # Generate structural autoencoder predictions if structure-tuning is enabled
        # These heads learn to reconstruct pre-computed structural tokens from sequence representations
        
        if self.use_evo_aae_heads:
            # EVOLUTIONARY ENVIRONMENT PREDICTION
            # Predict evolutionary environment class for each residue position
            # 
            # This is part of the structural autoencoder decoder:
            # - During preprocessing: 3D structure → evolutionary tokens (encoder, pre-computed)
            # - During training: sequence features → evolutionary tokens (decoder, learned)
            # 
            # Input: Contextual features (batch_size, seq_len-2, hidden_dim)  
            # Output: Logits over evolutionary classes (batch_size, seq_len-2, 513)
            pred['evo_logits'] = self.evo_head(feats)
            
            # ATOMIC ENVIRONMENT PREDICTION
            # Predict atomic environment class for each residue position
            # 
            # This is part of the structural autoencoder decoder:
            # - During preprocessing: 3D structure → atomic environment tokens (encoder, pre-computed)  
            # - During training: sequence features → atomic environment tokens (decoder, learned)
            #
            # Input: Contextual features (batch_size, seq_len-2, hidden_dim)
            # Output: Logits over atomic environment classes (batch_size, seq_len-2, 65)
            pred['aae_logits'] = self.aae_head(feats)
        else:
            # If not using structural heads, set to None
            # This happens during regular ESM2 training without structure-tuning
            pred['evo_logits'] = None
            pred['aae_logits'] = None
        
        return pred


# ===== ARCHITECTURAL NOTES =====
"""
Key Design Principles of ISM:

1. **Input Compatibility**: ISM maintains complete compatibility with ESM2 input format.
   No vocabulary expansion - same 33 tokens (20 amino acids + 13 special tokens).

2. **Multi-task Learning**: Three prediction tasks trained jointly:
   - Original MLM: Predict masked amino acids (33 classes)
   - Evolutionary: Predict evolutionary environment (512 classes) 
   - Atomic: Predict atomic environment (64 classes)

3. **Structural Autoencoder**: 
   - Encoder: 3D structure → discrete tokens (pre-computed during dataset creation)
   - Decoder: Sequence representations → structural tokens (learned during training)

4. **Shared Representations**: All prediction heads use the same contextual features
   from the ESM2 backbone, enabling transfer of structural knowledge to sequence understanding.

5. **Drop-in Replacement**: After structure-tuning, ISM can be used exactly like ESM2
   for downstream tasks, but with enhanced structural understanding.

Training Process:
1. Load pre-trained ESM2 backbone
2. Add structural prediction heads (evo_head, aae_head)
3. Train with combined loss: MLM + structural reconstruction
4. Result: ESM2 with enhanced structural representations

Usage Patterns:
- Structure-tuning: Use all heads with 'allmergedce' loss function
- Fine-tuning: Can freeze backbone, fine-tune heads for specific tasks
- Inference: Can use just the backbone features for downstream applications
"""
