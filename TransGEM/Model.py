import math
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.utils import softmax
from .gene_ex_encoder import *


def subsequent_mask(h, size_y, size_x):
    attn_shape = (h, size_y, size_x)
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

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)   

c = copy.deepcopy 
    
class TransGTM(nn.Module):
    def __init__(self, mode, args):
        super(TransGTM, self).__init__()
        self.param=args
        self.dim=args.hidden_dim
        self.pad_idx = args.pad_idx
        self.h=args.TF_H
        self.tgt_embedding = Embeddings(args.hidden_dim, args.vocab_size)
        if args.gene_encoder == "tenfold_binary":
            self.gene_embedding = nn.Linear(9+15,args.hidden_dim)
        elif args.gene_encoder == "binary":
            self.gene_embedding = nn.Linear(10+15,args.hidden_dim)
        elif args.gene_encoder == "one_hot":
            self.gene_embedding = nn.Linear(25+15,args.hidden_dim)
        elif args.gene_encoder == "value":
            self.gene_embedding = nn.Linear(1+15,args.hidden_dim)
        else:
            print("Please imput any one of [tenfold_binary, binary, one_hot, value]")
        self.position = c(PositionalEncoding(args.hidden_dim, args.PE_dropout))
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.hidden_dim,nhead=args.TF_H, dim_feedforward=args.ff_dim, batch_first=True, dropout=args.TF_dropout,activation= args.TF_act)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=args.TF_N)
        self.generator = Generator(args.hidden_dim, args.vocab_size)
        
        
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
            
            x = self.gene_embedding(x)
            x = self.position(x)
            y = data[i].y.t().squeeze(0)
            y_mask = (y[:-1] != self.pad_idx).unsqueeze(-2)
            y_mask = y_mask & Variable(subsequent_mask(self.h, y[:-1].size(-1), y[:-1].size(-1)).type_as(y_mask.data))
            y_mask = anti_bool(y_mask)
            y = self.tgt_embedding(y)
            y_p = self.position(y)

            y_pred = self.decoder(y_p[:,:-1], x, tgt_mask=y_mask)
            y_p = self.generator(y_pred)
            out.append(y_p)
        return out
            
        