"""
Transformer Seq2Seq Model for Code Generation
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer Encoder for NL understanding."""
    
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    
    def forward(self, src, src_mask=None):
        """src: (batch, seq_len)"""
        src_emb = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src_emb = self.pos_encoder(src_emb)
        output = self.transformer_encoder(src_emb, src_key_padding_mask=src_mask)
        return output


class TransformerDecoder(nn.Module):
    """Transformer Decoder for code generation."""
    
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        tgt: (batch, tgt_seq_len) - Decoder input
        memory: (batch, src_seq_len, d_model) - Encoder output
        """
        tgt_emb = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        output = self.transformer_decoder(
            tgt_emb, 
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        logits = self.fc_out(output)
        return logits


class Seq2SeqTransformer(nn.Module):
    """
    Complete Seq2Seq Transformer model for code generation.
    
    Architecture:
        - Encoder: Processes natural language description
        - Decoder: Generates code tokens
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=8, 
                 num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512, 
                 dropout=0.1, max_len=100):
        super().__init__()
        
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout
        )
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward, dropout
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for decoder."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    
    def forward(self, src, tgt, src_mask=None):
        """
        Forward pass for training.
        
        Args:
            src: (batch, src_len) - Source tokens (NL)
            tgt: (batch, tgt_len) - Target tokens (Code)
            src_mask: (batch, src_len) - Source padding mask
        
        Returns:
            logits: (batch, tgt_len, tgt_vocab_size)
        """
        # Encode source
        memory = self.encoder(src, src_mask)
        
        # Generate masks
        tgt_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        
        # Decode
        logits = self.decoder(
            tgt, 
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_mask
        )
        
        return logits
    
    def encode(self, src, src_mask=None):
        """Encode source sequence."""
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None, memory_key_padding_mask=None):
        """Decode one step."""
        return self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                          memory_key_padding_mask=memory_key_padding_mask)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model
    model = Seq2SeqTransformer(
        src_vocab_size=10000,
        tgt_vocab_size=10000,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256
    )
    
    batch_size = 4
    src_len = 20
    tgt_len = 15
    
    src = torch.randint(0, 10000, (batch_size, src_len))
    tgt = torch.randint(0, 10000, (batch_size, tgt_len))
    src_mask = torch.zeros(batch_size, src_len).bool()
    
    logits = model(src, tgt, src_mask)
    print(f"Model output shape: {logits.shape}")  # (batch, tgt_len, vocab_size)
    print(f"Parameters: {count_parameters(model):,}")
