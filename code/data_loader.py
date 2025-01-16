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


class DataSetCatCon(Dataset):
    def __init__(self, X, continuous_mean_std=None):
        X_mask = X['mask'].copy()
        X = X['data'].copy()

        self.X2 = X.copy().astype(np.float32)  # numerical columns
        self.X2_mask = X_mask.copy().astype(np.int64)  # numerical columns

        self.cls = np.zeros((X.shape[0],1), dtype=int)
        self.cls_mask = np.ones((X.shape[0],1), dtype=int)

        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return self.X2.shape[0]

    def __getitem__(self, idx):
        # X2 has continuous data
        return np.concatenate((self.cls[idx], self.X2[idx])), \
               np.concatenate((self.cls_mask[idx], self.X2_mask[idx]))


def shuffle_data(seed,features,info_df = None,indices = None):
    if indices is None:
        np.random.seed(seed)
        indices = np.random.permutation(features[0].shape[0])
    permut_features = []
    for i in range(len(features)):
        feature = features[i]
        permut_features.append(np.array([feature[j] for j in indices]))
    if info_df is not None:
        info_df = info_df.reindex(indices)
        info_df = info_df.reset_index(drop=True)
    return permut_features,info_df

def load_data_info(file_name):
    info_df = pd.read_csv("../data/{}_info.tsv".format(file_name),sep = "\t",header = 0)
    return info_df
