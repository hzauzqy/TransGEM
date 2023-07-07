import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import precision_recall_curve

import heapq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, DataListLoader, Batch
from torch_geometric.nn import DataParallel
from torch.autograd import Variable

import selfies as sf
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, MolFromSmiles, AllChem
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PandasTools
from rdkit.Chem import Draw
from rdkit import DataStructs

from TransGEM.utils import *
from TransGEM.gene_ex_encoder import *
from TransGEM.evaluation import *
from TransGEM.dataset import bulit_test_dataset
from TransGEM.Model import TransGTM

# Load SELFIES dictionary
TGT_stoi=np.load(args.data_path+"TGT_stoi.npy", allow_pickle=True).item()
TGT_itos=np.load(args.data_path+"TGT_itos.npy", allow_pickle=True)

# Set device
device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")

# Load test data
df=pd.read_csv(args.data_path+"{}_test.csv".format(args.dataset))

# Building test dataset
path = args.data_path+'test/'
ge_list=list(df["gene_e"])
cell_line=list(df["cell_line"])
sf_list=list(df["selfies"])
test_dataset=bulit_test_dataset(path, args.dataset,ge_list, cell_line=cell_line, sf_list=sf_list)

# Score of beam search generated molecules
def prob_score(probs, alpha):
    len_Y = len(probs)
    ln = (5 + len_Y)**alpha / (5 + 1)**alpha
    score = sum(torch.log(probs)) / ln
    return score

# beam search generated molecules
def out_decode(model, data, device, args):  
    y_pred = [torch.ones(1, 1).fill_(args.start_idx).type_as(data.y.data)]
    if args.gene_encoder == "tenfold_binary":
        x=torch.FloatTensor(tenfold_binary(data.x)).to(device)
    elif args.gene_encoder == "binary":
        x=torch.FloatTensor(binary(data.x)).to(device)
    elif args.gene_encoder == "one_hot":
        x=torch.FloatTensor(one_hot(data.x)).to(device)
    elif args.gene_encoder == "value":
        x=torch.FloatTensor(value(data.x)).to(device)
    x=torch.cat((x,data.cell_type.repeat(978,1)),1)
    x = model.gene_embedding(x)
    x = model.position(x)
    softmax=nn.Softmax(dim=1)
    for _ in range(args.max_len):
        all_seq=[]
        for i in y_pred:
            if _==0:
                y = model.tgt_embedding(i)
                y = model.position(y)
                out = model.decoder(y, x, None, None)
                prob = softmax(model.generator(out[:, -1])).detach()
                prob=prob.cpu()
                top_k_score, top_k_words=prob.view(-1).topk(args.k, dim=0,largest=True, sorted=True)
                i=i.cpu()
                out_=torch.cat([i.repeat(args.k,1), top_k_words.unsqueeze(1)], dim=1)
                prob_=top_k_score.unsqueeze(1)

                for j in range(len(prob_)):
                    n_s=out_[j].numpy().tolist().count(args.start_idx)
                    n_p=out_[j].numpy().tolist().count(args.pad_idx)
                    n_0=out_[j].numpy().tolist().count(0)
                    if n_s< 2 and n_p<1 and n_0<1:
                        all_seq.append((j,[prob_score(prob_[j], args.alpha), out_[j], prob_[j]]))
            else:
                if type(i) == tuple:
                    i=i[1]
                if args.end_idx in i[1].cpu().numpy().tolist():
                    all_seq.append(i)
                else:
                    y = model.tgt_embedding(i[1].unsqueeze(0).to(device))
                    y = model.position(y)
                    out = model.decoder(y, x, None, None)
                    prob = softmax(model.generator(out[:, -1])).detach()
                    prob = prob.cpu()
                    top_k_score, top_k_words=prob.view(-1).topk(args.k, dim=0,largest=True, sorted=True)
                    i[1], i[2] = i[1].cpu(), i[2].cpu()
                    out_=torch.cat([i[1].cpu().repeat(args.k,1), top_k_words.unsqueeze(1)], dim=1)
                    prob_=torch.cat([i[2].cpu().repeat(args.k,1), top_k_score.unsqueeze(1)], dim=1)

                    for j in range(len(prob_)):
                        n_s=out_[j].numpy().tolist().count(args.start_idx)
                        n_p=out_[j].numpy().tolist().count(args.pad_idx)
                        n_0=out_[j].numpy().tolist().count(0)
                        if n_s< 2 and n_p<1 and n_0<1:
                            all_seq.append([prob_score(prob_[j], args.alpha), out_[j], prob_[j]])

        if len(all_seq)<=args.seq_num:
            y_pred=all_seq
        else:
            y_pred=heapq.nlargest(args.seq_num, all_seq, key=lambda x:x[0])
    return y_pred

# model building
Model = TransGTM(args.dataset, args)
# load optimal parameters
ckpt = torch.load("ckpt/best_ckpt_{}".format(args.gene_encoder), map_location=torch.device('cpu'))
Model.load_state_dict(ckpt['model_state_dict'])
model = Model.to(device)
model.eval()

# model running
OUT=[]
print("\n########### Running in {} #############\n".format(device))

for i in range(len(test_dataset)):
    aa=[]
    print("sample{}".format(i))
    aa.append("sample{}".format(i))
    data = test_dataset[i].to(device)
    y_pred = out_decode(model, data, device, args)
    y_pred_list=[id2seq(j[1].numpy().tolist(), TGT_itos) for j in y_pred]
    aa.append(str(y_pred_list))
    out_valid, out_unval_index = valid(y_pred_list)
    aa.append(out_valid)
    out_valid_list=[]
    for j in range(len(y_pred_list)):
        if j not in out_unval_index:
            out_valid_list.append(y_pred_list[j])
    out_set, out_unique = unique(out_valid_list)
    aa.append(out_unique)
    out_div = IntDivp(out_set)
    aa.append(np.mean(out_div))
    qed_list, qed_scores = Qed(out_set)
    MW_list=[m[0] for m in qed_list]
    SA_scores = SA(out_set)
    aa.append(np.median(heapq.nlargest(10, qed_scores)))
    aa.append(str(heapq.nlargest(10, qed_scores)))
    aa.append(np.median(heapq.nlargest(10, MW_list)))
    aa.append(str(heapq.nlargest(10, MW_list)))
    SA_scores = SA(out_set)
    aa.append(np.median(heapq.nsmallest(10, SA_scores)))
    aa.append(str(heapq.nsmallest(10, SA_scores)))
    OUT.append(aa)
    
# Save test results
floder="./result"
if not os.path.exists(floder):
    os.makedirs(floder)
df = pd.DataFrame(OUT, columns=[
    "ID","preds","valid","unique","div","qed_mid","qed","mw_mid","mw","sa_mid","sa"])
df.to_csv("./result/test_{}_{}.csv".format(args.dataset, args.gene_encoder), index=False)
df.head()