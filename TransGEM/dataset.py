import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import random
from torch_geometric.data import Data, InMemoryDataset, Dataset
import torch_geometric.transforms as T
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, MolFromSmiles, AllChem

from torchtext.legacy import data
from torchtext.vocab import Vectors
from .utils import *

TGT_stoi=np.load(args.data_path+"TGT_stoi.npy", allow_pickle=True).item()
TGT_itos=np.load(args.data_path+"TGT_itos.npy", allow_pickle=True)

def load_dataset(path, dataset, train_path, val_path, teat_path):
    save_path = path + 'dataset_{}.ckpt'.format(dataset)
    if os.path.isfile(save_path):
        trn, val, test, max_len, vocab_size = torch.load(save_path)
        return trn, val, test, max_len, vocab_size
    
    df_train = pd.read_csv(os.path.join('{}'.format(train_path)))
    df_val = pd.read_csv(os.path.join('{}'.format(val_path)))
    df_test = pd.read_csv(os.path.join('{}'.format(teat_path)))
    
    selfies_train,cell_line_train,gene_e_train=list(df_train["selfies"]),list(df_train["cell_line"]),list(df_train["gene_e"])
    selfies_val, cell_line_val, gene_e_val = list(df_val["selfies"]), list(df_val["cell_line"]), list(df_val["gene_e"])
    selfies_test, cell_line_test, gene_e_test = list(df_test["selfies"]), list(df_test["cell_line"]), list(df_test["gene_e"])
    
    selfies_List = selfies_train + selfies_val + selfies_test
    cell_line = cell_line_train + cell_line_val + cell_line_test
    gene_e = gene_e_train + gene_e_val + gene_e_test
    if len(selfies_List) == len(gene_e):
        print("number of all data: ", len(selfies_List))
    cellline_List = cellline_encode(cell_line)  
    gene_List = [[float(j) for j in i.split("//")] for i in gene_e]
    target, max_len, vocab_size = selfies2id(selfies_List, TGT_stoi)
    
    mydataset = MyDataset(path, dataset, target, cellline_List, gene_List)

    trn, val, test = mydataset[:len(df_train)], \
                     mydataset[len(df_train):len(df_train)+len(df_val)], \
                     mydataset[len(df_train)+len(df_val):]
    
    torch.save([trn, val, test, max_len, vocab_size], save_path)
    return load_dataset(path, dataset, train_path, val_path, teat_path)


def bulit_test_visual_dataset(path, root, dataset, seed):
    save_path = root + '/{}.ckpt'.format(dataset)
    if os.path.isfile(save_path):
        dataset, max_len, vocab_size = torch.load(save_path)
        return dataset, max_len, vocab_size
    
    df = pd.read_csv(os.path.join('{}.csv'.format(path + dataset)))
    selfies_List, cell_line, gene_e = list(df["selfies"]), list(df["cell_line"]), list(df["gene_e"])
    dose = None
    if len(selfies_List) == len(gene_e):
        print("number of all data: ", len(selfies_List))
    cellline_List = cellline_encode(cell_line)
    gene_List = [[float(j) for j in i.split("//")] for i in gene_e]
    target, max_len, vocab_size = selfies2id(selfies_List)
    
    
    mydataset = MyDataset(root, dataset, target, dose, cellline_List, gene_List)

    
    torch.save([mydataset, max_len, vocab_size], save_path)
    return bulit_test_visual_dataset(path, root, dataset, seed)


def bulit_test_dataset(path, dataset, ge_list, cell_line=None, sf_list=None):
    
    print("number of all data: ", len(ge_list))
    gene_List = [[float(j) for j in i.split("//")] for i in ge_list]
    
    if cell_line is not None:
        cellline_List = cellline_encode(cell_line)
    else:
        cellline_List = None
        print("Please imput cell_type!")
    
    if sf_list is not None:
        target, max_len, vocab_size = selfies2id(sf_list, TGT_stoi)
    else:
        target=[[1]]*len(gene_List)
    
    mydataset = MyDataset(path, dataset, target, cellline_List, gene_List)
    
    return mydataset


class MyDataset(InMemoryDataset):

    def __init__(self, path, dataset, target, cellline_List, gene_List, transform=None, pre_transform=None, pre_filter=None):
        self.dataset = dataset
        self.target = target
        self.cellline_List = cellline_List   
        self.gene_List = gene_List
        super(MyDataset, self).__init__(path+dataset, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['{}.csv'.format(self.dataset)]
        pass

    @property
    def processed_file_names(self):
        return ['{}.pt'.format(self.dataset)]
        pass

    def download(self):
        pass

    def process(self):
        data_list = []

        for i in tqdm(range(len(self.target))):
#             cell_type=torch.FloatTensor([self.cellline_List[i]]).repeat(978,1)
#             gene_e=torch.FloatTensor(self.gege(self.gene_List[i]))
            data = Data(
                x=torch.FloatTensor(self.gene_List[i]),
                cell_type=torch.FloatTensor([self.cellline_List[i]]),
                y=torch.LongTensor(self.target[i]).unsqueeze(0).t()
            )
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



# def selfies2id(selfies_List):
#     selfiesID = []
#     src = []
#     src_len = []
#     for i in selfies_List:
#         i = i[2:-2]
#         src.append(i.split("', '"))
#         src_len.append(len(i.split("', '")))
#     max_len = max(src_len) + 2
#     TGT_stoi=np.load("./TGT_stoi.npy", allow_pickle=True).item()
#     for i in src:
#         a = []
#         a.append(TGT_stoi[SOS_WORD])
#         for j in i:
#             if j in list(TGT_stoi):
#                 a.append(TGT_stoi[j])
#             else:
#                 a.append(TGT_stoi[PAD_WORD])
#         a.append(TGT_stoi[EOS_WORD])
# #         while len(a) < max_len:
# #             a.append(TGT.vocab.stoi[BLANK_WORD])
#         selfiesID.append(a)

#     return np.array(selfiesID), max_len, len(TGT_stoi)

# # BOS_WORD = '<s>'
# # EOS_WORD = '</s>'
# # BLANK_WORD = "<blank>"
# PAD_WORD = '<pad>'
# UNK_WORD = '<unk>'
# EOS_WORD = '<eos>'
# SOS_WORD = '<sos>'
# MASK_WORD = '<mask>'
