import torch
import numpy as np


def embed_data_mask(x_cont, con_mask,model,vision_dset=False):
    device = x_cont.device
    # x_categ = x_categ + model.categories_offset.type_as(x_categ)
    # x_categ_enc = model.embeds(x_categ)
    n1,n2 = x_cont.shape
    # _, n3 = x_categ.shape
    if model.cont_embeddings == 'MLP':
        x_cont_enc = torch.empty(n1,n2, model.dim)
        for i in range(model.num_continuous):
            # print(x_cont[:,i].shape)
            x_cont_enc[:,i,:] = model.simple_MLP[i](x_cont[:,i])
            # print(x_cont_enc[:,i,:].shape)
    else:
        raise Exception('This case should not work!')

    x_cont_enc = x_cont_enc.to(device)
    con_mask_temp = con_mask + model.con_mask_offset.type_as(con_mask)
    con_mask_temp = model.mask_embeds_cont(con_mask_temp)
    x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]

    return x_cont_enc


def mixup_data(x1, x2 , lam=1.0, y= None, use_cuda=True):
    '''Returns mixed inputs, pairs of targets'''

    batch_size = x1.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)


    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    if y is not None:
        y_a, y_b = y, y[index]
        return mixed_x1, mixed_x2, y_a, y_b
    
    return mixed_x1, mixed_x2


def add_noise(x_cont, noise_params = {'noise_type' : ['cutmix'],'lambda' : 0.1}):
    lam = noise_params['lambda']
    device = x_cont.device
    batch_size = x_cont.size()[0]

    index = torch.randperm(batch_size)
    # cat_corr = torch.from_numpy(np.random.choice(2,(x_categ.shape),p=[lam,1-lam])).to(device)
    con_corr = torch.from_numpy(np.random.choice(2,(x_cont.shape),p=[lam,1-lam])).to(device)
    x2 = x_cont[index,:]
    x_cont_corr = x_cont.clone().detach()
    # x_categ_corr[cat_corr==0] = x1[cat_corr==0]
    x_cont_corr[con_corr==0] = x2[con_corr==0]
    return x_cont_corr

