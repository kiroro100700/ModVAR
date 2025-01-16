import torch
from torch import nn

from data_openml import DataSetCatCon
from torch.utils.data import DataLoader
import torch.optim as optim
from tab_pretrain.augmentations import embed_data_mask


def SAINT_pretrain(model,X_train,device):
    pt_aug = ['cutmix']
    pt_tasks = ['contrastive','denoising']
    pt_projhead_style = 'diff'
    mixup_lam = 0.3
    nce_temp = 0.7
    lam0 = 0.5

    train_ds = DataSetCatCon(X_train)
    trainloader = DataLoader(train_ds, batch_size=64, shuffle=True)

    # vision_dset = opt.vision_dset
    optimizer = optim.AdamW(model.parameters(),lr=0.0001)
    pt_aug_dict = {
        'noise_type': ['cutmix'],
        'lambda': 0.1
    }
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    print("Pretraining begins!")
    for epoch in range(100):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            x_cont, con_mask = data[0].to(device), data[1].to(device)
            x_cont = x_cont.float()
            con_mask = con_mask.int()
            # embed_data_mask function is used to embed both categorical and continuous data.
            if 'cutmix' in pt_aug:
                from tab_pretrain.augmentations import add_noise
                x_cont_corr = add_noise(x_cont, noise_params = pt_aug_dict)
                x_cont_enc_2 = embed_data_mask(x_cont_corr, con_mask,model)
            else:
                # print("no cutmix")
                x_cont_enc_2 = embed_data_mask(x_cont, con_mask,model)
            x_cont_enc = embed_data_mask(x_cont, con_mask,model)

            loss = 0
            if 'contrastive' in pt_tasks:
                aug_features_1 = model.transformer(x_cont_enc)
                aug_features_2 = model.transformer(x_cont_enc_2)
                aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1,2)
                aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1,2)
                # print(aug_features_2.shape)
                if pt_projhead_style == 'diff':
                    aug_features_1 = model.pt_mlp(aug_features_1)
                    aug_features_2 = model.pt_mlp2(aug_features_2)
                    # print(aug_features_2.shape)
                elif pt_projhead_style == 'same':
                    aug_features_1 = model.pt_mlp(aug_features_1)
                    aug_features_2 = model.pt_mlp(aug_features_2)
                else:
                    print('Not using projection head')

                logits_per_aug1 = aug_features_1 @ aug_features_2.t()/nce_temp
                logits_per_aug2 =  aug_features_2 @ aug_features_1.t()/nce_temp
                targets = torch.arange(logits_per_aug1.size(0)).to(logits_per_aug1.device)
                loss_1 = criterion1(logits_per_aug1, targets)
                loss_2 = criterion1(logits_per_aug2, targets)
                loss = lam0*(loss_1 + loss_2)/2

            if 'denoising' in pt_tasks:
                con_outs = model(x_cont_enc_2)
                if len(con_outs) > 0:
                    con_outs = torch.cat(con_outs,dim=1)
                    l2 = criterion2(con_outs, x_cont)
                else:
                    l2 = 0
                l1 = 0
                loss += l1 + l2

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch: {epoch}, Running Loss: {running_loss}')

    print('END OF PRETRAINING!')
    return model