import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import precision_recall_curve

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import DataParallel
from torch_geometric.data import DataLoader, DataListLoader, Batch


class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))



class Trainer():
    def __init__(self, model, train_dataset, valid_dataset, test_dataset, device, args, records=None):
        
        self.device = device
        self.model = model.to(self.device)
        self.args = args
        
        if records is None:
            self.records = {'trn_record': [], 'val_record': [], 'val_losses': [], 'train_losses': [],'best_ckpt': None}
        else:
            self.records = records
        
        """Setting the train valid and test data loader"""
        self.train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size)
        self.test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        """Setting the optimizer with hyper-param"""
        self.criterion = LabelSmoothing(size=args.vocab_size, padding_idx=args.pad_idx, smoothing=0.1).to(self.device)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.7, patience=6, min_lr=1e-6)

        self.start = time.time()
        
        self.log('train set num:{}    valid set num:{}    test set num: {}'.format(
            len(train_dataset), len(valid_dataset), len(test_dataset)))
        self.log("total parameters:" + str(sum([p.nelement() for p in self.model.parameters()])))
        self.log(msgs=str(model).split('\n'), show=False)

    def train_iterations(self):
        self.model.train()
        losses = []
        for data in tqdm(self.train_dataloader):
            self.optimizer.zero_grad()
            data = data.to(self.device)
            out_pred = self.model(data, self.device)               
            loss = 0
            for i in range(len(out_pred)):
                loss+=self.criterion(out_pred[i].squeeze(0), data[i].y[1:].t().squeeze(0))/len(out_pred)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
           
        trn_loss = np.array(losses).mean()

        return trn_loss

    def valid_iterations(self):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for data in tqdm(self.valid_dataloader):
                data = data.to(self.device)
                out_pred = self.model(data, self.device)               
                loss = 0
                for i in range(len(out_pred)):
                    loss+=self.criterion(out_pred[i].squeeze(0), data[i].y[1:].t().squeeze(0))/len(out_pred)
                losses.append(loss.item())
        val_loss = np.array(losses).mean()

        return val_loss

    def train(self):
        floder = self.args.out_path+"ckpts_{}/{}/dim{}_n{}h{}ff{}_bh{}_lr{}".format(self.args.dataset,self.args.gene_encoder,
            self.args.hidden_dim,self.args.TF_N,self.args.TF_H,self.args.ff_dim,self.args.batch_size,self.args.lr)
        if not os.path.exists(floder):
            os.makedirs(floder)
  
        self.log('Training start({}_{}_dim{}_n{}h{}ff{}_bh{}_lr{})'.format(
            self.args.dataset,self.args.gene_encoder,self.args.hidden_dim,self.args.TF_N,self.args.TF_H,
            self.args.ff_dim,self.args.batch_size,self.args.lr))
        early_stop_cnt = 0
        for epoch in range(self.args.epochs):
            
            trn_loss = self.train_iterations()
            val_loss = self.valid_iterations()

            self.scheduler.step(val_loss)
            lr_cur = self.scheduler.optimizer.param_groups[0]['lr']

            self.log('Epoch:{} trn_loss:{:.5f} lr_cur:{:.5f}'.format(epoch, trn_loss, lr_cur), with_time=True)
            self.log('Epoch:{} val_loss:{:.5f} lr_cur:{:.5f}'.format(epoch, val_loss, lr_cur), with_time=True)

            self.records['val_losses'].append(val_loss)
            self.records['train_losses'].append(trn_loss)
            self.records['val_record'].append([epoch+1, val_loss, lr_cur])
            self.records['trn_record'].append([epoch+1, trn_loss, lr_cur])
            if val_loss == np.array(self.records['val_losses']).min():
                self.save_model_and_records(epoch)
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1

        self.save_model_and_records(epoch, final_save=True)

    def save_model_and_records(self, epoch, final_save=False):
        if final_save:
            self.save_loss_records()
        else:
            self.records['best_ckpt'] = "{}_best_model.ckpt".format(self.args.dataset)

        with open(self.args.out_path+"ckpts_{}/{}/dim{}_n{}h{}ff{}_bh{}_lr{}/ckpt{}".format(
            self.args.dataset,self.args.gene_encoder,self.args.hidden_dim,self.args.TF_N,self.args.TF_H,
            self.args.ff_dim,self.args.batch_size,self.args.lr, epoch), 'wb') as f:
            torch.save({'records': self.records,'model_state_dict': self.model.state_dict(),}, f)
        self.log('Model saved at epoch {}'.format(epoch))

    def save_loss_records(self):
        if not os.path.exists(self.args.out_path+"ckpts_{}/{}/recoders".format(self.args.dataset,self.args.gene_encoder)):
            os.makedirs(self.args.out_path+"ckpts_{}/{}/recoders".format(self.args.dataset,self.gene_encoder))
        trn_record = pd.DataFrame(self.records['trn_record'],columns=['epoch', 'trn_loss', 'lr'])
        val_record = pd.DataFrame(self.records['val_record'],columns=['epoch', 'val_loss', 'lr'])
        ret = pd.DataFrame({
            'epoch': trn_record['epoch'],
            'trn_loss': trn_record['trn_loss'],
            'val_loss': val_record['val_loss'],
            'trn_lr': trn_record['lr'],
            'val_lr': val_record['lr']
        })
        ret.to_csv(self.args.out_path+'ckpts_{}/{}/recoders/record_dim{}_n{}h{}ff{}_bh{}_lr{}.csv'.format(
            self.args.dataset,self.args.gene_encoder,self.args.hidden_dim,self.args.TF_N,self.args.TF_H,
            self.args.ff_dim,self.args.batch_size,self.args.lr), index=False)
        return ret

    def log(self, msg=None, msgs=None, with_time=False, show=True):
        if with_time: msg = msg + ' time elapsed {:.2f} hrs ({:.1f} mins)'.format(
            (time.time() - self.start) / 3600.,
            (time.time() - self.start) / 60.
        )
        if not os.path.exists(self.args.out_path+"ckpts_{}/{}/log".format(self.args.dataset,self.args.gene_encoder)):
            os.makedirs(self.args.out_path+"ckpts_{}/{}/log".format(self.args.dataset,self.args.gene_encoder))
        with open(self.args.out_path+'ckpts_{}/{}/log/log_dim{}_n{}h{}ff{}_bh{}_lr{}.txt'.format(
            self.args.dataset,self.args.gene_encoder,self.args.hidden_dim,self.args.TF_N,
            self.args.TF_H,self.args.ff_dim,self.args.batch_size,self.args.lr), 'a+') as f:
            if msgs:
                self.log('#' * 80)
                if '\n' not in msgs[0]: msgs = [m + '\n' for m in msgs]
                f.writelines(msgs)
                if show:
                    for x in msgs:
                        print(x, end='')
                self.log('#' * 80)
            if msg:
                f.write(msg + '\n')
                if show:
                    print(msg)
