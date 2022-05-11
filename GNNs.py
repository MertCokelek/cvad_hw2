import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import utils


class MLP(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super(MLP, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class Sub_Graph_Block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Sub_Graph_Block, self).__init__()
        self.g_enc = MLP(in_dim, out_dim)

    def enc(self, x):
        return self.g_enc(x)

    def agg(self, x, mask):
        out = torch.max(x.masked_fill(mask == 0, -1e12), dim=1).values.unsqueeze(1)
        return out

    def rel(self, x_enc, x_agg):
        return torch.cat([x_enc, x_agg], dim=2)

    def forward(self, x, mask):
        x_enc = self.enc(x)
        x_agg = self.agg(x_enc, mask)
        x_agg = x_agg.repeat(1, x.shape[1], 1)
        out = self.rel(x_enc, x_agg)
        return out


class Self_Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Self_Attention, self).__init__()

        self.Q = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.K = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.V = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, attention_mask=None):
        P_q = self.Q(x)
        P_k = self.K(x)
        P_v = self.V(x)

        QKT = torch.bmm(P_q, P_k.permute(0, 2, 1))
        if attention_mask is not None:
            QKT = QKT.masked_fill(attention_mask.to(QKT.device) == 0, -1e12)
        out = torch.bmm(self.softmax(QKT), P_v)
        return out


class Global_Graph(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = Self_Attention(hidden_size)

    def forward(self, hidden_states, attention_mask=None, mapping=None):
        out = self.attn(hidden_states, attention_mask)

        return out


class Sub_Graph(nn.Module):
    def __init__(self, hidden_size, depth=3):
        super(Sub_Graph, self).__init__()
        self.blocks = nn.ModuleList(Sub_Graph_Block(hidden_size, hidden_size // 2) for _ in range(depth))

    def agg(self, x):
        out = torch.max(x, dim=1).values
        return out  # .unsqueeze(1)

    def forward(self, hidden_states, lengths):
        lengths = torch.Tensor(lengths).cuda()
        n_nodes, n_vecs, n_hidden = hidden_states.shape

        mask = torch.zeros(n_nodes, n_vecs, n_hidden//2).cuda()
        mask_idx = torch.arange(n_vecs).repeat(n_nodes, 1).cuda()

        out = hidden_states
        for block in self.blocks:
            mask_idx_ = mask_idx.unsqueeze(2).repeat(1, 1, n_hidden//2)
            mask[mask_idx_ < lengths.unsqueeze(1).unsqueeze(1)] = 1

            out = block(out, mask)
        out = self.agg(out)
        return out
