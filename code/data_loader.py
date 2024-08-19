import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd

#DataLoader
class MyDataset(Dataset):
    def __init__(self, dna_fea,aa_fea, tool_fea, anno_fea, epi_fea,pc_fea,y=None):
        self.dna_fea = dna_fea
        self.aa_fea = aa_fea
        self.tool_fea = tool_fea
        self.anno_fea = anno_fea
        self.pc_fea = pc_fea
        self.epi_fea = epi_fea
        self.y = None
        if y != None:
            self.y = y

        self.tab_fea = torch.concat([self.anno_fea,self.pc_fea,self.epi_fea],dim=1)
        index = torch.isnan(self.tab_fea)
        nan_mask = torch.ones(self.tab_fea.shape, dtype=torch.int)
        nan_mask[index] = 0
        self.nan_mask = nan_mask
        self.cls = torch.zeros((self.tab_fea.shape[0],1), dtype=torch.int)
        self.cls_mask = torch.ones((self.tab_fea.shape[0],1), dtype=torch.int)

    def __len__(self):
        return self.dna_fea.shape[0]

    def __getitem__(self, idx):
        tab_ = torch.concat((self.cls[idx],self.tab_fea[idx]))
        mask_ = torch.concat((self.cls_mask[idx],self.nan_mask[idx]))
        if self.y!=None:
            return self.dna_fea[idx],self.aa_fea[idx],self.tool_fea[idx],tab_,mask_,self.y[idx]
        else:
            return self.dna_fea[idx], self.aa_fea[idx], self.tool_fea[idx], tab_, mask_


def shuffle_data(seed,features,info_df = None):
    np.random.seed(seed)
    indices = np.random.permutation(features[0].shape[0])
    permut_features = []
    for i in range(len(features)):
        feature = features[i]
        permut_features.append(np.array([feature[j] for j in indices]))
    if info_df is not None:
        info_df = info_df.reindex(indices)
    return permut_features,info_df

def load_data_info(file_name):
    info_df = pd.read_csv("../data/{}_info.tsv".format(file_name),sep = "\t",header = 0)
    return info_df
