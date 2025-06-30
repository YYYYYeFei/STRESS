import torch.utils.data as data
import torch
import numpy as np

class MyDataset(data.Dataset):
    def __init__(self,arg,samples):
        # print('load data...')
        self.samples = samples
        self.arg=arg
        self.HR=[]
        self.LR = []
        for s in self.samples:
            if self.arg.mode not in ['train','test']:
                self.HR.append(self.arg.data_root + f"/HR/{s}_HR.npy")
            else:
                if arg.sampling == 'all':
                    LR_w=['even_1','even_2','odd_1','odd_1']
                    self.LR.append([self.arg.data_root + f"/LR_{l}/{s}_LR_{l}.npy" for l in LR_w])
                    self.HR.append(self.arg.data_root + f"/HR/{s}_HR.npy")
                else:
                    self.LR.append(self.arg.data_root + f"/LR_{arg.sampling}/{s}_LR_{arg.sampling}.npy")
                    self.HR.append(self.arg.data_root + f"/HR/{s}_HR.npy")

    def __len__(self):
        return len(self.HR)

    def __getitem__(self, idx):
        HR = np.load(self.HR[idx]).transpose(2,1,0)[:self.arg.gene_num]
        GT = torch.FloatTensor(HR).unsqueeze(0)
        if self.arg.mode in ['train','test']:
            if self.arg.sampling == 'all':
                LR=[]
                for i in range(len(self.LR[idx])):
                    temp=np.load(self.LR[idx][i]).transpose(2,1,0)[:self.arg.gene_num,:,:]
                    LR.append(temp)
                LR=np.array(LR)
                img_trans = torch.FloatTensor(LR)
            else:
                LR = np.load(self.LR[idx]).transpose(2, 1, 0)[:self.arg.gene_num, :, :]
                img_trans = torch.FloatTensor(LR).unsqueeze(0)
            data=tuple((img_trans,GT))
        else:
            data = GT
        return data




