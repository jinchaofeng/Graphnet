import torch.nn as nn
from graphattention_layer import GraphAttentionLayer


class GCN_unit(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GCN_unit, self).__init__()
        self.gat = GraphAttentionLayer(dim_in, dim_out, 0, 0.2)
        self.gcn11 = nn.Linear(dim_out, dim_out, bias=False)
        self.act = nn.ReLU()
        self.downsample = nn.Conv1d(dim_in, dim_out, 1)

    def forward(self, A, x):
        h = self.gat(x, A)
        x1 = A @ h
        X1 = self.gcn11(x1)
        X1 = self.act(X1)
        x = x.permute(0, 2, 1)
        x = self.downsample(x)
        x = x.permute(0, 2, 1)
        out = X1 + x
        return out


class model_GCN(nn.Module):
    def __init__(self):
        super(model_GCN, self).__init__()
        self.g_unit1 = GCN_unit(1, 16)
        self.g_unit2 = GCN_unit(16, 32)
        self.g_unit3 = GCN_unit(32, 16)
        self.g_unit4 = GCN_unit(16, 3)

    def forward(self, A, x):
        x1 = self.g_unit1(A, x)
        x2 = self.g_unit2(A, x1)
        x3 = self.g_unit3(A, x2)
        x4 = self.g_unit4(A, x3)
        return x4
