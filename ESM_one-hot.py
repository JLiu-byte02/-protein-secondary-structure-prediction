import os
import torch
import torch.nn as nn
import esm
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ESM model
model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
model = model.to(device).eval()
batch_converter = alphabet.get_batch_converter()

# One-hot encoding
valid_aa = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M',
            'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X']
valid_dict = {k: i for i, k in enumerate(valid_aa)}
def encode_one_hot(seq):
    idxs = [valid_dict.get(res, valid_dict['X']) for res in seq]
    return np.eye(len(valid_aa))[idxs]  #(L, 21)

# Secondary structure labels
ss_list = ['C', 'B', 'E', 'G', 'I', 'H', 'S', 'T', ' ']
ss_index_dict = {ss: idx for idx, ss in enumerate(ss_list)}

# Read labels
ss_path = r"/root/sequence.ss"
ss_dict = {}
with open(ss_path, "r") as f:
    for line in f:
        if ',' not in line:
            continue
        seq, ss = line.strip().split(',')
        ss_dict[seq] = ss


# Transformer encoder
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=4, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, max_len=1000):
        super(TransformerEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


# Initialization and Transformer encoder
project_layer = nn.Linear(2560, 512).to(device)
transformer_module = TransformerEncoder(num_layers=4, d_model=512).to(device)

# Output file
out_path = "/root/sequence.feat"
with open(out_path, 'w') as w:
    for seq, ss in ss_dict.items():
        data = [("seq", seq)]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[36], return_contacts=False)
        rep = results["representations"][36][0, 1:len(seq)+1]  # (L, 2560)

        # Dimension reduction + Transformer
        rep_proj = project_layer(rep).unsqueeze(0)  # (1, L, 512)
        transformer_out = transformer_module(rep_proj).squeeze(0)  # (L, 512)

        # Linear mapping to 21 dimensions
        head = nn.Linear(512, 21).to(device)
        with torch.no_grad():
            pred_feat = head(transformer_out)  #(L, 21)

        # One-hot concatenation
        onehot_21d = torch.tensor(encode_one_hot(seq), dtype=torch.float32, device=device)  # (L, 21)
        combined = torch.cat([pred_feat, onehot_21d], dim=-1)  # (L, 42)

        # Padding to 700 rows
        combined = combined.detach().cpu().numpy()
        pad_len = 700 - len(seq)
        if pad_len > 0:
            combined = np.pad(combined, ((0, pad_len), (0, 0)), mode='constant')

        # Write feature count
        w.write("700\n")

        # Write feature vectors
        for row in combined:
            w.write(' '.join(map(str, row)) + '\n')

        # Write labels
        for ch in ss:
            w.write(f"{ss_index_dict.get(ch, 8)}\n")
        for _ in range(pad_len):
            w.write("8\n")
