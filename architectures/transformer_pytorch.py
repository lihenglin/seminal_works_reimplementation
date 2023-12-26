# TODO: For now, only think about arhicecture and don't worry about underlying computation like FlashAttention

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math


class MHA(nn.Module):

    def __init__(self, d_model=512, num_heads=8):
        super(MHA, self).__init__()
        self.num_heads = num_heads
        self.scale = torch.sqrt(torch.tensor(d_model // num_heads, dtype=torch.float32))

        self.qproj = nn.Linear(d_model, d_model)
        self.kvproj = nn.Linear(d_model, 2 * d_model)
        self.finalproj = nn.Linear(d_model, d_model)

    def forward(self, x, y=None, mask=None):
        if y is None:
            y = x

        # Projection
        Q = self.qproj(x)
        K, V = einops.rearrange(self.kvproj(y), 'B L (2 d_model) -> B L d_model 2').unbind(dim=-1)

        # Rearrange to multi-head Q, K, V
        Q = einops.rearrange(Q, 'B L (num_heads d_k) -> B num_heads L d_k', num_heads=self.num_heads)
        K = einops.rearrange(K, 'B L (num_heads d_k) -> B num_heads L d_k', num_heads=self.num_heads)
        V = einops.rearrange(V, 'B L (num_heads d_v) -> B num_heads L d_v', num_heads=self.num_heads)

        # Scaled Dot Product Attention
        score = einops.einsum(Q, K, 'B num_heads L1 d_k, B num_heads L2 d_k -> B num_heads L1 L2') / self.scale
        if mask is not None: # (L, L)
            score += mask
        attn = torch.softmax(score, dim=-1)
        attn_v = einops.einsum(attn, V, 'B num_heads L1 L2, B num_heads L2 d_v -> B num_heads L1 d_v')
        attn_v = einops.rearrange(attn_v, 'B num_heads L d_v -> B L (num_heads d_v)')
        return self.finalproj(attn_v)


class TransformerEncoderBlock(nn.Module):
    
    def __init__(self, d_model=512, num_heads=8, dropout_rate=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.dropout_rate = dropout_rate

        self.mha = MHA(d_model, num_heads)
        self.mha_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * d_model, d_model)
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Pre-Norm architecture is less sensitive to learning rate warmup
        x = x + F.dropout(self.mha(self.mha_norm(x)), p=self.dropout_rate)
        x = x + F.dropout(self.ffn(self.ffn_norm(x)), p=self.dropout_rate) 
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model=512, max_seq_len=1024):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_seq_len).unsqueeze(1) # (L, 1), unsqueeze for broadcasting
        div_term = torch.exp(-torch.arange(0, d_model, 2) / d_model * math.log(10000.0)) # (d_model / 2)
        pe = torch.zeros(1, max_seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1), :]


class TransformerEncoder(nn.Module):

    def __init__(self, d_model=512, num_heads=8, dropout_rate=0.1, num_classes=10, max_seq_len=1024):
        super(TransformerEncoder, self).__init__()
        self.dropout_rate = dropout_rate
        self.emb_scale = math.sqrt(d_model)

        self.emb_lookup = nn.Embedding(num_embeddings=256, embedding_dim=d_model) # Just a placeholder. Can have different ways to get the embeddings for different applications
        self.pos_emb_lookup = PositionalEncoding(d_model, max_seq_len)
        self.transformer = nn.Sequential(
            *[TransformerEncoderBlock(d_model, num_heads, dropout_rate) for _ in range(6)]
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        emb = F.dropout(self.emb_lookup(x) * self.emb_scale + self.pos_emb_lookup(x), p=self.dropout_rate)
        return self.classifier(self.transformer(emb))


class TransformerDecoderBlock(nn.Module):
    
    def __init__(self):
        super(TransformerDecoderBlock, self).__init__()
        pass

    def forward(self, x):
        pass


class TransformerDecoder(nn.Module):
    
    def __init__(self):
        super(TransformerDecoder, self).__init__()
        pass

    def forward(self, x):
        pass




if __name__ == '__main__':
    # TODO: [Test] use enwik9 and monitor loss
    pass