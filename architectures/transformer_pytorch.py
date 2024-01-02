import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
from tqdm import tqdm


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
        K, V = einops.rearrange(self.kvproj(y), 'B L (split d_model) -> B L d_model split', split=2).unbind(dim=-1)

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

    def forward(self, x, mask=None):
        # Pre-Norm architecture is less sensitive to learning rate warmup
        x = x + F.dropout(self.mha(self.mha_norm(x), mask=mask), p=self.dropout_rate)
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
        self.transformer = nn.ModuleList(
            [TransformerEncoderBlock(d_model, num_heads, dropout_rate) for _ in range(6)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        emb = F.dropout(self.emb_lookup(x) * self.emb_scale + self.pos_emb_lookup(x), p=self.dropout_rate) # NOTE: I can't find a reasonable explanation for self.emb_scale, but seems to be what people usually do
        for layer in self.transformer:
            emb = layer(emb, mask=mask)
        return self.classifier(self.norm(emb))


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


class LM(nn.Module):
    
    def __init__(self, num_classes=10, max_seq_len=1024):
        super(LM, self).__init__()
        self.transformer = TransformerEncoder(num_classes=num_classes, max_seq_len=max_seq_len)
        self.register_buffer('mask', torch.triu(torch.full((max_seq_len, max_seq_len), float('-inf')), diagonal=1))

    def forward(self, x):
        x = torch.cat((255 * torch.ones_like(x[:, 0:1]), x[:, :-1]), dim=1) # NOTE: 0xFF never occur in enwik9
        return self.transformer(x, mask=self.mask)


### Just for testing purpose ###
""" Borrow from here (https://github.com/hengyuan-hu/jax-vs-pytorch) """
import numpy as np
import random

class Enwik9Loader:
    """Iterator that returns shuffled slices of Enwik9"""

    def __init__(self, batch_size: int, seq_len: int, datapath: str):
        self.arr = np.fromfile(datapath, dtype=np.uint8)
        self.batch_size = batch_size
        self.seq_len = seq_len

    def __iter__(self):
        # Make slice boundaries randomized across epochs
        offset = random.randint(0, self.seq_len - 1)
        offset_len = self.arr.size - offset
        seqs = offset_len // self.seq_len
        slices = np.array(
            [
                self.arr[start : start + self.seq_len]
                for start in range(offset, offset + seqs * self.seq_len, self.seq_len)
            ]
        )
        np.random.default_rng().shuffle(slices)
        short_batch = len(slices) % self.batch_size
        batches = [
            slices[start : start + self.batch_size]
            for start in range(0, len(slices) - short_batch, self.batch_size)
        ]
        return iter(batches)

if __name__ == '__main__':
    dataloader = list(Enwik9Loader(batch_size=100, seq_len=256, datapath='/iliad/u/lhlin/seminal_works_reimplementation/datasets/enwik9'))
    num_classes, max_seq_len = 256, 256

    LM = LM(num_classes, max_seq_len).cuda()
    optimizer = torch.optim.Adam(LM.parameters(), lr=3e-4)
    # TODO: linear learning rate decay
    
    loss_fn = nn.CrossEntropyLoss()

    avg_loss = 0.
    for i, batch in tqdm(enumerate(dataloader)):
        batch = torch.tensor(batch, dtype=torch.long).cuda()
        loss = loss_fn(einops.rearrange(LM(batch), 'B L C -> B C L'), batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        if i % 100 == 0:
            print(f"Loss: {avg_loss / 100}")
            avg_loss = 0.

    # TODO: low precision training