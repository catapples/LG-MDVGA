import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from autils import *


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, drop=0.4, bias=False, activation=None):
        super(GraphConv, self).__init__()
        self.dropout = nn.Dropout(drop)
        self.activation = activation
        self.w = nn.Linear(in_dim, out_dim, bias=bias)
        self.w2 = nn.Linear(out_dim, out_dim, bias=bias)
        self.cin = in_dim + out_dim + out_dim * 3
        self.w3 = nn.Linear(self.cin, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.w.weight)
        self.bias = bias
        if self.bias:
            nn.init.zeros_(self.w.bias)

    def forward(self, adj, x):
        x = self.dropout(x)
        x1 = self.w(torch.matrix_power(adj, 0).cuda().mm(x))
        for k in range(0, 2):
            z = self.w(torch.matrix_power(adj, k).cuda().mm(x))
            x1 = x1 + z
        if self.activation:
            return self.activation(x1)
        else:
            return x1


class VGAEModel(nn.Module):
    def __init__(self, in_dim, hidden1_dim):
        super(VGAEModel, self).__init__()

        self.in_dim = in_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = in_dim
        layers = [GraphConv(self.in_dim, self.hidden1_dim, activation=F.relu),
                  GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x),  # lambda x: x
                  GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x),
                  GraphConv(self.in_dim, self.in_dim, activation=F.relu)]
        self.layers = nn.ModuleList(layers)

    def encoder(self, g, features):
        h = self.layers[0](g, features)

        mean = self.layers[1](g, h)
        log_std = self.layers[2](g, h)
        gaussian_noise = torch.randn(features.size(0), self.hidden2_dim).cuda()
        sampled_z = mean + gaussian_noise * torch.exp(log_std).cuda()
        return sampled_z, mean, log_std, h

    def forward(self, g, features):
        z, _, _, _ = self.encoder(g, features)
        z1 = torch.sigmoid(z)
        return z1


class Parameter(nn.Module):
    def __init__(self, f):
        super(Parameter, self).__init__()
        self.lnc_e = nn.Parameter(nn.init.xavier_uniform_(torch.empty(f.shape[0], 1)), requires_grad=True)
        self.pro_e = nn.Parameter(nn.init.xavier_uniform_(torch.empty(f.shape[1], 1)), requires_grad=True)

    def forward(self):
        Ilnce = torch.mm(self.lnc_e, self.lnc_e.T)  # .detach().numpy()
        Iproe = torch.mm(self.pro_e, self.pro_e.T)  # .detach().numpy()
        Ilnce = F.softmax(F.relu(Ilnce), dim=1).detach().numpy()
        Iproe = F.softmax(F.relu(Iproe), dim=1).detach().numpy()

        return Ilnce, Iproe


class DGCN(nn.Module):
    def __init__(self, out_dim, hid_dim, bias=False):
        super(DGCN, self).__init__()
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.res1 = GraphConv(out_dim, hid_dim, bias=bias, activation=F.relu)
        self.res2 = GraphConv(hid_dim, hid_dim, bias=bias, activation=torch.tanh)
        self.res3 = GraphConv(hid_dim, hid_dim, bias=bias, activation=F.relu)
        self.res4 = GraphConv(hid_dim, out_dim, bias=bias, activation=torch.sigmoid)

    def forward(self, g, z):
        z = self.res2(g,self.res1(g,z))
        res = self.res4(g,self.res3(g,z))
        # z = self.res1(g, z)
        # res = self.res4(g, z)

        return res