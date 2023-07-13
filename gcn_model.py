from __future__ import print_function
from __future__ import division


# import libraries
import sys
import pickle as pkl
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy import stats
import torch
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from utils import normalize_torch
import ipdb

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.normal_(tensor=self.weight)
        if self.bias is not None:
            torch.nn.init.normal_(tensor=self.bias)
    
    def forward(self, inp, adj):
        support = torch.mm(inp, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat_list, dropout, nblock, adj, nlay, num_edges,num_nodes,sym):
        super(GCN, self).__init__()      
        assert len(nfeat_list)==nlay+1
        self.nlay = nlay
        self.sym=sym
        self.nblock = nblock
        self.num_edges = num_edges
        self.num_nodes = num_nodes
        self.dropout = dropout
        gcs_list = []

        for i in range(nlay):
            if i==0:
                gcs_list.append([str(i), GraphConvolution(nfeat_list[i], nfeat_list[i+1])])
            else:
                gcs_list.append([str(i), GraphConvolution(nfeat_list[i], nfeat_list[i+1])])

    
        self.gcs = nn.ModuleDict(gcs_list)
        self.nfeat_list = nfeat_list
    
    def forward(self, x, labels, adj, nz_idx, obs_idx, adj_normt, training=True):

        for i in range(self.nlay):
            x = self.gcs[str(i)](x, adj)
            
            if i != self.nlay-1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=training)

                
        output = F.log_softmax(x, dim=1) 
        nll_loss = self.loss(labels, output, obs_idx)

        return output, nll_loss
    
    def loss(self, labels, preds, obs_idx):
        return F.nll_loss(preds[obs_idx], labels[obs_idx])