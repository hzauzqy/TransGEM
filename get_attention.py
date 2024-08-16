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
df=pd.read_csv(args.data_path+"data_com_gene.csv")
df_lmg=pd.read_csv(args.data_path+"978gene.csv")
ge_list=list(df["gene_e"])
cell_line=list(df["cell_line"])
sf_list=list(df["selfies"])

# Building test dataset
path = args.data_path+'test/'
test_dataset=bulit_test_dataset(path, args.dataset, ge_list, cell_line=cell_line, sf_list=sf_list)

# model building
Model = TransGTM(args.dataset, args)
# load optimal parameters
ckpt = torch.load("ckpt/best_ckpt_{}".format(args.gene_encoder), map_location=torch.device('cpu'))
Model.load_state_dict(ckpt['model_state_dict'])
model = Model.to(device)
model.eval()


# 设置mask函数
def subsequent_mask(size_n, size_d):
    attn_shape = (8, size_n, size_d)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
    
# 设置提取Attention函数
def get_features_hook(module, data_input, data_output):
    a, b = [], []
    a.append(data_output)
    b.append(data_input)
    input_list.append(b)
    att_list.append(a)
    
# 设置模型运行时提取Attention矩阵
for i in range(args.TF_N):
    model.decoder.layers[i].multihead_attn.register_forward_hook(hook=get_features_hook)
    
# 加载基因信息
data_g=pd.read_csv(args.data_path+"geneinfo.csv")
gid = list(data_g["gene_id"])
gsb = list(data_g["gene_symbol"])
esi=list(data_g["ensembl_id"])


# 加载化合物及对应相互作用基因信息
data1=pd.read_excel(args.data_path+"com_gene.xlsx")
com_g=list(data1["gene"])
sm1=list(data1["smiles"])


# 加载试验数据
data2=pd.read_csv(args.data_path+"data_com_gene.csv")
sm2=list(data2["smiles"])
cell_l=list(data2["cell_line"])
data2.head()

# 运行模型并得到每个sample对应的Attention基因排名
OUt=[]
for ii in range(len(test_dataset)):
    b=[]
    data=test_dataset[ii]
    genes=com_g[sm1.index(sm2[ii])][2:-2].split("', '")
    IND1=[]
    for i in genes:
        a=[]
        i=i.split("&&")
        ind1=gid.index(int(i[0]))
        ind2=gsb.index(i[1])
        if ind1==ind2:
            IND1.append(ind1)
        else:
            print(i)
    data = data.to(device)
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
    y = data.y[:-1].t().squeeze(0)
    y_mask = (y != args.pad_idx).unsqueeze(-2)
    y_mask = y_mask & Variable(subsequent_mask(y.size(-1), y.size(-1)).type_as(y_mask.data))
    y_mask = anti_bool(y_mask)
    y = model.tgt_embedding(y)
    y = model.position(y)
    att_list, input_list = [], []
    model.decoder(y, x, tgt_mask=y_mask)
    out=torch.zeros(att_list[0][0][1][0].size(0), att_list[0][0][1][0].size(1))
    for i in range(6):
        out+=att_list[i][0][1][0].cpu()
    out_att=out[-1]
    for i in range(len(out)-1):
        out_att+=out[i]
    t = copy.deepcopy(list(out_att.detach()))
    top, top_ind = [], []
    for _ in range(978):
        index = t.index(max(t))
        top.append(max(t))
        top_ind.append(index)
        t[index] = 0
    ranks=[]
    for d in IND1:
        ranks.append(top_ind.index(d))
    a1,a2,a3,a4,a5=0,0,0,0,0
    for r in ranks:
        if r <=10:
            a1+=1
        if r <=20:
            a2+=1
        if r <=50:
            a3+=1
        if r <=100:
            a4+=1
        if r <=500:
            a5+=1
    print("##############################################################")
    print("Num{}".format(ii),"\t",cell_l[ii], "\t", sm2[ii], "\n", "all:", len(IND1), "\t",
          "top10:{}, top20:{}, top50:{}, top100:{}, top500:{}".format(a1,a2,a3,a4,a5), "\n")
    
    b.extend([cell_l[ii],sm2[ii],str(genes),str(ranks),a1,a2,a3,a4,a5,len(ranks)])
    OUt.append(b)
    
# 保存结果
dff=pd.DataFrame(OUt, columns=["cell_line", "smiles", "genes", "rank", "top10", "top20", "top50", "top100", "top500", "number"])
dff.to_csv("result/com_gene_rank.csv")