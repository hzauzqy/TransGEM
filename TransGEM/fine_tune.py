import math
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.utils import softmax

from .Model import TransGTM
from .gene_ex_encoder import *


def subsequent_mask(size_y, size_x):
    attn_shape = (8, size_y, size_x)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def anti_bool(mask):
    return mask == 0

def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False
            
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(0)], 
                         requires_grad=False)
        return self.dropout(x)

c = copy.deepcopy 

class TransGTM_ft(nn.Module):
    def __init__(self, model, args):
        super(TransGTM_ft, self).__init__()
        self.param=args
        self.dim=args.hidden_dim
        self.pad_idx = args.pad_idx
        self.TransGEM = model
        self.fine = nn.Linear(args.hidden_dim,args.hidden_dim)
        freeze(self.TransGEM)
        
    def forward(self, data, device):

        out = []
        for i in range(len(data.ptr)-1):
            if self.param.gene_encoder == "tenfold_binary":
                x=torch.FloatTensor(tenfold_binary(data[i].x)).to(device)
            elif self.param.gene_encoder == "binary":
                x=torch.FloatTensor(binary(data[i].x)).to(device)
            elif self.param.gene_encoder == "one_hot":
                x=torch.FloatTensor(one_hot(data[i].x)).to(device)
            elif self.param.gene_encoder == "value":
                x=torch.FloatTensor(value(data[i].x)).to(device)
            else:
                print("Please imput any one of [tenfold_binary, binary, one_hot, value]")
            x=torch.cat((x,data[i].cell_type.repeat(978,1)),1)
            x = self.TransGEM.gene_embedding(x)
     
            x = self.TransGEM.position(x)
            y = data[i].y.t().squeeze(0)
            y_mask = (y[:-1] != self.pad_idx).unsqueeze(-2)
            y_mask = y_mask & Variable(subsequent_mask(y[:-1].size(-1), y[:-1].size(-1)).type_as(y_mask.data))
            y_mask = anti_bool(y_mask)
            y = self.TransGEM.tgt_embedding(y)
            y_p = self.TransGEM.position(y)

            y_pred = self.TransGEM.decoder(y_p[:,:-1], x, tgt_mask=y_mask)
            y_pred = self.fine(y_pred)
            y_p = self.TransGEM.generator(y_pred)
            out.append(y_p)
        return out