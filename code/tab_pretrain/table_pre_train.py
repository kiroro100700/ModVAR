import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pretraining import SAINT_pretrain
from pretrainmodel import SAINT
import torch
from torch import nn
from data_openml import DataSetCatCon
from torch.utils.data import DataLoader
import torch.optim as optim
from tab_pretrain.augmentations import embed_data_mask

file_name = "COSMIC_5000"
fea_mat = np.load('../../数据集/注释文件/gene_anno_{}.npy'.format(file_name))
print(fea_mat.shape)

pc_fea = pd.read_csv('../../数据集/PanCanAtlas/{}_wzStatic_100bpx2_pancan_fea.tsv'.format(file_name),sep = "\t",header = 0)
tobe_drop = ["t_depth", "t_ref_count", "t_alt_count", "n_depth", "n_ref_count", "n_alt_count"]
pc_fea.drop(tobe_drop,axis = 1,inplace = True)
pc_mat = pc_fea.to_numpy()
print(pc_mat.shape)

epi_df = pd.read_hdf("../../数据集/TVAR特征/{}_info_96.fea".format(file_name))
epi_df.drop('chr',axis = 1,inplace = True)
epi_df.drop('pos',axis = 1,inplace = True)
epi_df.drop('ref',axis = 1,inplace = True)
epi_df.drop('alt',axis = 1,inplace = True)
epi_mat = epi_df.to_numpy()
print(epi_mat.shape)

tab_fea = np.concatenate([fea_mat,pc_mat],axis=1)
tab_fea = np.concatenate([tab_fea,epi_mat],axis=1)
print(tab_fea.shape)

index = np.isnan(tab_fea)
nan_mask = np.ones(tab_fea.shape)
nan_mask[index] = 0

X_train = {'data':tab_fea,'mask':nan_mask}
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}.")

#pretrain
model = SAINT(
num_continuous = X_train['data'].shape[1]+1,
dim = 4,
dim_out = 1,
depth = 1,
heads = 4,
attn_dropout = 0.8,
ff_dropout = 0.8,
mlp_hidden_mults = (4, 2),
cont_embeddings = "MLP",
attentiontype = "colrow",
final_mlp_style = "sep",
y_dim = 2
)
model.to(device)
model.float()
model = SAINT_pretrain(model,X_train,device)

torch.save(model,'../tab_model/saint_100epoch.pth')

#load
# model = torch.load('../tab_model/saint_1000epoch.pth')
# model.eval()
# model.to(device)
# # print(X_train['data'][0].shape)
# test_ds = DataSetCatCon(X_train)
# print(test_ds[:,0])
# testloader = DataLoader(test_ds, batch_size=64, shuffle=True)
#
# for i, data in enumerate(testloader, 0):
#     x_cont, con_mask = data[0].to(device), data[1].to(device)
#     print(x_cont.shape)
#     x_cont = x_cont.float()
#     con_mask = con_mask.int()
#     x_cont_enc = embed_data_mask(x_cont, con_mask, model)
#     print(x_cont_enc.shape)
#     output = model.transformer(x_cont_enc)
#     # print(output[:,0,:])
#     print(output[:,0,:].shape)
#     break

