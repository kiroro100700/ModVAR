import fea_loader
from model import ModVAR
from data_loader import MyDataset,shuffle_data,load_data_info,DataSetCatCon
from fea_extract import encode_dna_seq,encode_aa_seq
from pretrainmodel import SAINT
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torch import optim
from sklearn.metrics import precision_recall_curve,roc_curve,auc
from sklearn.metrics import f1_score
import sys
import argparse
import os
import pandas as pd
from scipy import stats


cuda_is_available = torch.cuda.is_available()
print("cuda:",cuda_is_available)

def set_seed(seed):
    torch.manual_seed(seed)
    if cuda_is_available:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def rightness(predictions,labels):
    pred = torch.round(predictions)
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights,len(labels)

def count_mean_and_error(x):
    x_np = np.array(x)
    # print("mean：",x_np.mean(axis=0))
    # print("error：",np.std(x_np))
    return (x_np.mean(axis=0), np.std(x_np))

def Metrics(label,scores):
    # auprc
    precision, recall, thresholds = precision_recall_curve(label, scores)
    auprc = auc(recall, precision)
    # auroc
    fp, tp, thresholds_ = roc_curve(label, scores)
    auroc = auc(fp, tp)
    # f1 & Youden metrics
    j_scores = tp - fp
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds_[best_idx]
    # # mean_score = np.mean(scores)
    predictions = [1 if scores[i] >= best_threshold else 0 for i in range(scores.shape[0])]
    f1 = f1_score(label, predictions)
    #p_value
    x0_all,x1_all = [],[]
    for idx in range(len(label)):
        if label[idx] == 0:
            x0_all.append(predictions[idx])
        else:
            x1_all.append(predictions[idx])
    statistic, pvalue = stats.mannwhitneyu(x0_all, x1_all, use_continuity=True, alternative='two-sided')
    return auroc,auprc,f1,pvalue


def calculate_metrics(tool_name,file_name,sample = True):
    if file_name == "civic":
        tool_roc = []
        tool_prc = []
        tool_f1 = []
        tool_p_value = []
        for i in range(10):
            neg_file = "val_COSMIC_neg_{}".format(i)
            pos_df = fea_loader.load_compare_score("val_civic",tool_name)
            neg_df = fea_loader.load_compare_score(neg_file,tool_name,benchmark_dir=True)

            pos_df = pos_df[pos_df!="."].astype(float)
            pos_np = pos_df.to_numpy()
            neg_df = neg_df[neg_df != "."].astype(float)
            neg_np = neg_df.to_numpy()
            label = [1 for _ in range(pos_np.shape[0])]+[0 for _ in range(neg_np.shape[0])]
            scores = np.concatenate([pos_np,neg_np],axis=0)

            # auprc
            precision, recall, thresholds = precision_recall_curve(label, scores)
            auprc = auc(recall, precision)
            # auroc
            fp, tp, thresholds_ = roc_curve(label, scores)
            auroc = auc(fp, tp)
            #f1 & Youden metrics
            j_scores = tp - fp
            best_idx = np.argmax(j_scores)
            best_threshold = thresholds_[best_idx]
            # # mean_score = np.mean(scores)
            predictions = [1 if scores[i] >= best_threshold else 0 for i in range(scores.shape[0]) ]
            f1 = f1_score(label, predictions)
            # p_value
            x0_all, x1_all = [], []
            for idx in range(len(label)):
                if label[idx] == 0:
                    x0_all.append(predictions[idx])
                else:
                    x1_all.append(predictions[idx])
            statistic, pvalue = stats.mannwhitneyu(x0_all, x1_all, use_continuity=True, alternative='two-sided')

            # print('{} ROUND{}: AUPRC:{:.4f}, AUROC:{:.4f},f1:{:.4f}'.format(tool_name,i , auprc, auroc,f1))
            tool_roc.append(auroc)
            tool_prc.append(auprc)
            tool_f1.append(f1)
            tool_p_value.append(pvalue)
        #mean and error
        roc_np = np.array(tool_roc)
        prc_np = np.array(tool_prc)
        f1_np = np.array(tool_f1)
        p_value_np = np.array(tool_p_value)

        print("-"*100)
        print(f"Final result of {tool_name}:")
        print("ROC:{}±{}".format(roc_np.mean(axis=0),np.std(roc_np)))
        print("PRC:{}±{}".format(prc_np.mean(axis=0),np.std(prc_np)))
        print("F1:{}±{}".format(f1_np.mean(axis=0),np.std(f1_np)))
        # print("Pvalue:{}±{}".format(p_value_np.mean(axis=0),np.std(p_value_np)))
        return roc_np, prc_np,f1_np,p_value_np
    else:
        tool_roc = []
        tool_prc = []
        tool_f1 = []
        tool_p_value = []
        if "23" in file_name and sample:
            score_df = fea_loader.load_compare_score(file_name, tool_name)
            label = fea_loader.load_labels(file_name)
            pos_idx = [i for i, label in enumerate(label) if label == 1]
            neg_idx = [i for i, label in enumerate(label) if label == 0]
            sample_idx = []
            for i in range(3):
                sample_idx.append([pos_idx[i] for i in range(i, len(pos_idx), 3)]+neg_idx)

            for s_idx in sample_idx:
                label_ = [label[i] for i in s_idx]
                score_df_ = score_df.iloc[s_idx].copy()
                score_df_.reset_index(drop=True,inplace=True)

                idxs = score_df_[score_df_ != "."].index
                score_df_ = score_df_[score_df_ != "."].astype(float)
                scores_ = score_df_.to_numpy()
                label_ = [label_[i] for i in idxs]
                auroc, auprc, f1,p_value = Metrics( label_,scores_)
                tool_roc.append(auroc)
                tool_prc.append(auprc)
                tool_f1.append(f1)
                tool_p_value.append(p_value)
            # mean and error
            roc_np = np.array(tool_roc)
            prc_np = np.array(tool_prc)
            f1_np = np.array(tool_f1)
            p_value_np = np.array(tool_p_value)

            print("-" * 100)
            print(f"Final result of {tool_name}:")
            print("ROC:{}±{}".format(roc_np.mean(axis=0), np.std(roc_np)))
            print("PRC:{}±{}".format(prc_np.mean(axis=0), np.std(prc_np)))
            print("F1:{}±{}".format(f1_np.mean(axis=0), np.std(f1_np)))
            # print("pvalue:{}±{}".format(p_value_np.mean(axis=0), np.std(p_value_np)))

            return roc_np,prc_np,f1_np,p_value_np
        else:
            score_df = fea_loader.load_compare_score(file_name, tool_name)
            label = fea_loader.load_labels(file_name)

        ori_len = score_df.shape[0]

        idxs = score_df[score_df != "."].index
        score_df = score_df[score_df != "."].astype(float)
        scores = score_df.to_numpy()
        label = [label[i] for i in idxs]

        auroc,auprc,f1,p_value = Metrics(label,scores)

        tool_roc.append(auroc)
        tool_prc.append(auprc)
        tool_f1.append(f1)
        tool_p_value.append(p_value)

        # mean and error
        roc_np = np.array(tool_roc)
        prc_np = np.array(tool_prc)
        f1_np = np.array(tool_f1)
        p_value_np = np.array(tool_p_value)
        print("-" * 100)
        print(f"Final result of {tool_name}:")
        print("Cover:{}/{}".format(len(idxs),ori_len))
        print("ROC:{}".format(roc_np.mean(axis=0)))
        print("PRC:{}".format(prc_np.mean(axis=0)))
        print("F1:{}".format(f1_np.mean(axis=0)))
        # print("Pvalue:{}".format(p_value_np.mean(axis=0)))

        return roc_np.mean(axis=0),prc_np.mean(axis=0),f1_np.mean(axis=0),p_value_np.mean(axis=0)


def embed_data_mask(x_cont,con_mask,model):
    device = x_cont.device
    n1,n2 = x_cont.shape
    x_cont_enc = torch.empty(n1,n2, model.dim)

    for i in range(model.num_continuous):
        x_cont_enc[:,i,:] = model.simple_MLP[i](x_cont[:,i])

    x_cont_enc = x_cont_enc.to(device)
    con_mask_temp = con_mask + model.con_mask_offset.type_as(con_mask)
    con_mask_temp = model.mask_embeds_cont(con_mask_temp)
    x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]

    return x_cont_enc


def add_noise(x_cont, noise_params = {'noise_type' : ['cutmix'],'lambda' : 0.1}):
    lam = noise_params['lambda']
    device = x_cont.device
    batch_size = x_cont.size()[0]

    index = torch.randperm(batch_size)
    con_corr = torch.from_numpy(np.random.choice(2,(x_cont.shape),p=[lam,1-lam])).to(device)
    x2 = x_cont[index,:]
    x_cont_corr = x_cont.clone().detach()
    x_cont_corr[con_corr==0] = x2[con_corr==0]
    return x_cont_corr


def SAINT_pretrain(model, X_train, device,save_name = '',num_epoch = 500,batch_size=64):
    pt_aug = ['cutmix']
    pt_tasks = ['contrastive', 'denoising']
    pt_projhead_style = 'diff'
    nce_temp = 0.7
    lam0 = 0.5

    train_ds = DataSetCatCon(X_train)
    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # vision_dset = opt.vision_dset
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    pt_aug_dict = {
        'noise_type': ['cutmix'],
        'lambda': 0.1
    }
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    print("Pretraining begins!")
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            x_cont, con_mask = data[0].to(device), data[1].to(device)
            x_cont = x_cont.float()
            con_mask = con_mask.int()
            if 'cutmix' in pt_aug:
                x_cont_corr = add_noise(x_cont, noise_params=pt_aug_dict)
                x_cont_enc_2 = embed_data_mask(x_cont_corr, con_mask, model)
            else:
                # print("no cutmix")
                x_cont_enc_2 = embed_data_mask(x_cont, con_mask, model)
            x_cont_enc = embed_data_mask(x_cont, con_mask, model)

            loss = 0
            if 'contrastive' in pt_tasks:
                aug_features_1 = model.transformer(x_cont_enc)
                aug_features_2 = model.transformer(x_cont_enc_2)
                aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1, 2)
                aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1, 2)
                if pt_projhead_style == 'diff':
                    aug_features_1 = model.pt_mlp(aug_features_1)
                    aug_features_2 = model.pt_mlp2(aug_features_2)
                elif pt_projhead_style == 'same':
                    aug_features_1 = model.pt_mlp(aug_features_1)
                    aug_features_2 = model.pt_mlp(aug_features_2)
                else:
                    print('Not using projection head')

                logits_per_aug1 = aug_features_1 @ aug_features_2.t() / nce_temp
                logits_per_aug2 = aug_features_2 @ aug_features_1.t() / nce_temp
                targets = torch.arange(logits_per_aug1.size(0)).to(logits_per_aug1.device)
                loss_1 = criterion1(logits_per_aug1, targets)
                loss_2 = criterion1(logits_per_aug2, targets)
                loss = lam0 * (loss_1 + loss_2) / 2

            if 'denoising' in pt_tasks:
                con_outs = model(x_cont_enc_2)
                if len(con_outs) > 0:
                    con_outs = torch.cat(con_outs, dim=1)
                    l2 = criterion2(con_outs, x_cont)
                else:
                    l2 = 0
                l1 = 0
                loss += l1 + l2

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch: {epoch}, Running Loss: {running_loss}')
        if (epoch+1) % 100 == 0 and save_name != '':
            if not os.path.exists("../data/tab_model/{}".format(save_name)):
                os.mkdir("../data/tab_model/{}".format(save_name))
            torch.save(model, "../data/tab_model/{}/{}_epoch{}.pth".format(save_name, save_name, epoch + 1))

    print('END OF PRETRAINING!')
    if not os.path.exists("../data/tab_model/{}".format(save_name)):
        os.mkdir("../data/tab_model/{}".format(save_name))
    torch.save(model, "../data/tab_model/{}/{}_finish.pth".format(save_name, save_name))

    return model


def tab_model_pretrain(file_name = "train",save_name='',num_epoch =500,batch_size = 64):
    gene_mat,epi_mat,pc_mat = fea_loader.load_tabular_fea(file_name)
    tab_fea = np.concatenate([gene_mat, pc_mat], axis=1)
    tab_fea = np.concatenate([tab_fea, epi_mat], axis=1)
    print(tab_fea.shape)

    index = np.isnan(tab_fea)
    nan_mask = np.ones(tab_fea.shape)
    nan_mask[index] = 0

    X_train = {'data': tab_fea, 'mask': nan_mask}
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}.")

    # pretrain
    model = SAINT(
        num_continuous=X_train['data'].shape[1] + 1,
        dim=4,
        dim_out=1,
        depth=1,
        heads=4,
        attn_dropout=0.8,
        ff_dropout=0.8,
        mlp_hidden_mults=(4, 2),
        cont_embeddings="MLP",
        attentiontype="colrow",
        final_mlp_style="sep",
        y_dim=2
    )
    model.to(device)
    model.float()
    model = SAINT_pretrain(model, X_train, device,save_name=save_name,num_epoch=num_epoch,batch_size=batch_size)
    return model


def load_eval_set(val_name,neg_file = "val_COSMIC_neg_0",sample=True):
    if "civic" in val_name:
        pos_val = list(fea_loader.load_fea("val_civic","infer"))
        neg_val = list(fea_loader.load_COSMIC_neg_fea(neg_file,"infer"))
        features_val = []
        for i in range(6):
            features_val.append(np.concatenate((pos_val[i],neg_val[i]),axis = 0))
        features_val.append(pos_val[-1]+neg_val[-1])

        pos_info_df = load_data_info("val_civic")
        neg_info_df = load_data_info(neg_file)
        info_df = pd.concat((pos_info_df,neg_info_df),axis=0)
        info_df.reset_index(drop=True,inplace=True)
    else:
        features_val = list(fea_loader.load_fea(val_name, "infer"))
        info_df = load_data_info(val_name)
    return features_val,info_df


def evaluation(features_val,info_df,tab_model = "saint_fn.pth",model_name = "ModVAR.pth",output = None,seed = 15):
    features_val, info_df = shuffle_data(0, features_val, info_df)

    dna_valid = torch.DoubleTensor(features_val[0])
    aa_valid = torch.DoubleTensor(features_val[1])
    aa_valid = torch.where(torch.isnan(aa_valid), torch.zeros_like(aa_valid), aa_valid)
    gene_valid = torch.DoubleTensor(features_val[2])
    epis_valid = torch.DoubleTensor(features_val[3])
    pc_valid = torch.DoubleTensor(features_val[4])
    tool_valid = torch.DoubleTensor(features_val[5].astype(np.float64))
    label_valid = torch.DoubleTensor(features_val[6])


    val_set = MyDataset(dna_valid, aa_valid, tool_valid, gene_valid, epis_valid, pc_valid, label_valid)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False)

    model = torch.load('../data/{}'.format(model_name), map_location='cpu')
    model = model.double()
    tab_model = torch.load('../data/tab_model/{}'.format(tab_model), map_location='cpu')

    if cuda_is_available:
        model = model.cuda()
        tab_model = tab_model.cuda()

    model.eval()
    tab_model.eval()
    print("Evaluating...")
    val_loss = 0
    correct = 0
    prob_all = []
    label_all = []
    set_seed(seed)
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            print("\r{}".format(i), end="")
            dna_v = data[0]
            aa_v = data[1]
            tool_v = data[2]
            tab_v = data[3]
            mask_v = data[4]
            y_v = data[5]

            dna_v = dna_v.double()
            aa_v = torch.DoubleTensor(aa_v)
            tool_v = torch.DoubleTensor(tool_v)
            y_v = torch.DoubleTensor(y_v)

            if cuda_is_available:
                y_v = y_v.cuda()
                dna_v = dna_v.cuda()
                tool_v = tool_v.cuda()
                aa_v = aa_v.cuda()
                tab_v = tab_v.cuda()
                mask_v = mask_v.cuda()
            tab_v_enc = embed_data_mask(tab_v, mask_v, tab_model)

            # print(tab_v_enc)
            tab_v_out = tab_model.transformer(tab_v_enc)
            tab_v_out = tab_v_out.double()
            # print(tab_v_out)
            y_hat = model(dna_v, aa_x=aa_v, tool_x=tool_v, tab_x=tab_v_out)
            prob = y_hat.max(1).values
            prob_all.extend(prob.cpu().detach().numpy())
            label_all.extend(y_v.cpu().detach())

            right, counts = rightness(prob, y_v)
            correct += right
    # auprc
    precision, recall, thresholds = precision_recall_curve(label_all, prob_all)
    auprc = auc(recall, precision)

    # auroc
    fp, tp, thresholds_ = roc_curve(label_all, prob_all)
    auroc = auc(fp, tp)

    # f1 & Youden metrics
    j_scores = tp - fp
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds_[best_idx]
    # # mean_score = np.mean(scores)
    predictions = [1 if prob_all[i] >= best_threshold else 0 for i in range(len(prob_all))]
    f1 = f1_score(label_all, predictions)

    #p-value
    x0_all,x1_all = [],[]
    for idx in range(len(label_all)):
        if label_all[idx] == 0:
            x0_all.append(predictions[idx])
        else:
            x1_all.append(predictions[idx])
    statistic, pvalue = stats.mannwhitneyu(x0_all, x1_all, use_continuity=True, alternative='two-sided')

    # accuracy,Loss
    accuracy = 100. * correct / dna_valid.shape[0]
    print('Test set: Accuracy: {}/{} ({:.3f}%), AUPRC:{:.4f}, AUROC:{:.4f},F1:{:.4f}\n'.format(
        correct, dna_valid.shape[0], accuracy, auprc, auroc, f1))
    # print('Test set: Best thre:',best_threshold)
    # print('Test set: P_value:',pvalue,'\n')
    if output is not None:
        print("Writing the results into output...")
        with open("../output/{}.tsv".format(output), "w+") as f:
            f.write("Chr\tPos\tRef\tAlt\tLabel\tScore\n")
            for i in range(len(label_all)):
                line = '{}\t{}\t{}\t{}\t{}\t{:.6f}\n'.format(info_df.iloc[i, 0], info_df.iloc[i, 1], info_df.iloc[i, 2],
                                                             info_df.iloc[i, 3], label_all[i], prob_all[i])
                f.write(line)
        print("Successfully write into {}.".format(output))
    return accuracy.cpu(), auprc, auroc, f1,best_threshold,pvalue


def train_and_eval_ModVAR(train_name,val_name,tab_model = "saint_fn",batch_size = 32,n_epoch=20,save_name='',seed=15):
    features_train = list(fea_loader.load_fea(train_name,"infer"))
    features_train_1 = []
    features_train_2 = []
    for feature in features_train:
        features_train_1.append(feature[0:4246])
        features_train_2.append(feature[4246:])
    permut_features_train_1,_ = shuffle_data(seed, features_train_1)
    permut_features_train_2,_ = shuffle_data(seed, features_train_2)
    features_train = []
    for i in range(len(features_train_1)):
        features_train.append(np.concatenate((permut_features_train_1[i],permut_features_train_2[i]),axis = 0))

    # features_train,_ = shuffle_data(seed,features_train)
    dna_train = torch.DoubleTensor(features_train[0])
    aa_train = torch.DoubleTensor(features_train[1])
    aa_train = torch.where(torch.isnan(aa_train), torch.zeros_like(aa_train), aa_train)
    gene_train = torch.DoubleTensor(features_train[2])
    epis_train = torch.DoubleTensor(features_train[3])
    pc_train = torch.DoubleTensor(features_train[4])
    tool_train = torch.DoubleTensor(features_train[5].astype(np.float64))
    label_train = torch.DoubleTensor(features_train[6])

    train_set = MyDataset(dna_train, aa_train, tool_train, gene_train, epis_train, pc_train, label_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)


    features_val = list(fea_loader.load_fea(val_name,"infer"))
    features_val,_ = shuffle_data(0, features_val)


    dna_valid = torch.DoubleTensor(features_val[0])
    aa_valid = torch.DoubleTensor(features_val[1])
    aa_valid = torch.where(torch.isnan(aa_valid), torch.zeros_like(aa_valid), aa_valid)
    gene_valid = torch.DoubleTensor(features_val[2])
    epis_valid = torch.DoubleTensor(features_val[3])
    pc_valid = torch.DoubleTensor(features_val[4])
    tool_valid = torch.DoubleTensor(features_val[5].astype(np.float64))
    label_valid = torch.DoubleTensor(features_val[6])

    val_set = MyDataset(dna_valid, aa_valid, tool_valid, gene_valid, epis_valid, pc_valid, label_valid)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    set_seed(seed)

    model = ModVAR()
    model = model.double()
    tab_model = torch.load('../data/tab_model/{}'.format(tab_model), map_location='cpu')

    if cuda_is_available:
        model = model.cuda()
        tab_model = tab_model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    cost = nn.BCELoss()

    losses = []
    val_losses = []
    val_acc = []
    rights = []
    train_acc = []

    print("train")
    for epoch in range(n_epoch):
        correct = 0
        print("Training..Epoch:{}".format(epoch+1))
        for i, data in enumerate(train_loader):
            print("\r{}".format(i), end="")
            dna_t = data[0]
            aa_t = data[1]
            tool_t = data[2]
            tab_t = data[3]
            mask_t = data[4]
            y_t = data[5]

            dna_t = dna_t.double()
            aa_t = torch.DoubleTensor(aa_t)
            tool_t = torch.DoubleTensor(tool_t)
            y_t = torch.DoubleTensor(y_t)

            if cuda_is_available:
                y_t = y_t.cuda()
                dna_t = dna_t.cuda()
                tool_t = tool_t.cuda()
                aa_t = aa_t.cuda()
                tab_t = tab_t.cuda()
                mask_t = mask_t.cuda()
            tab_t_enc = embed_data_mask(tab_t, mask_t, tab_model)
            model.train()
            tab_model.eval()
            optimizer.zero_grad()

            tab_t_out = tab_model.transformer(tab_t_enc)
            tab_t_out = tab_t_out.double()
            y_hat = model(dna_x=dna_t, aa_x=aa_t, tool_x=tool_t, tab_x=tab_t_out)
            predict = torch.squeeze(y_hat, dim=1)
            loss = cost(predict, y_t)
            loss.backward()
            optimizer.step()
            losses.append(loss.data)

            y_hat = model(dna_x=dna_t, aa_x=aa_t, tool_x=tool_t, tab_x=tab_t_out)
            right, counts = rightness(y_hat, y_t)
            rights.append(right)
            correct += right

            if (i + 1) % 30 == 0:
                print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % (
                epoch + 1, n_epoch, i + 1, dna_train.shape[0] // batch_size, loss.item()))
        accuracy = 100. * correct / dna_train.shape[0]
        print('Stage 0: Epoch: {}, Loss: {:.5f}, Training set accuracy: {}/{} ({:.3f}%)'.format(epoch + 1, losses[-1], correct,
                                                                                       dna_train.shape[0], accuracy))
        model.eval()
        tab_model.eval()

        print("Evaluating...")
        val_loss = 0
        correct = 0
        prob_all = []
        label_all = []
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                print("\r{}".format(i), end="")
                optimizer.zero_grad()
                dna_v = data[0]
                aa_v = data[1]
                tool_v = data[2]
                tab_v = data[3]
                mask_v = data[4]
                y_v = data[5]

                dna_v = dna_v.double()
                aa_v = torch.DoubleTensor(aa_v)
                tool_v = torch.DoubleTensor(tool_v)
                y_v = torch.DoubleTensor(y_v)


                if cuda_is_available:
                    y_v = y_v.cuda()
                    dna_v = dna_v.cuda()
                    tool_v = tool_v.cuda()
                    aa_v = aa_v.cuda()
                    tab_v = tab_v.cuda()
                    mask_v = mask_v.cuda()
                tab_v_enc = embed_data_mask(tab_v, mask_v, tab_model)
                optimizer.zero_grad()

                tab_v_out = tab_model.transformer(tab_v_enc)
                tab_v_out = tab_v_out.double()
                y_hat = model(dna_x=dna_v, aa_x=aa_v, tool_x=tool_v, tab_x=tab_v_out)
                prob = y_hat.max(1).values
                prob_all.extend(prob.cpu().detach().numpy())
                label_all.extend(y_v.cpu().detach())

                loss = cost(prob, y_v)  # batch average loss
                loss = loss.data
                right, counts = rightness(prob, y_v)
                correct += right
                val_loss += loss * len(y_v)  # sum up batch loss
        # auprc
        precision, recall, thresholds = precision_recall_curve(label_all, prob_all)
        auprc = auc(recall, precision)

        # auroc
        fp, tp, thresholds_ = roc_curve(label_all, prob_all)
        auroc = auc(fp, tp)

        # accuracy,Loss
        val_losses.append(val_loss / dna_valid.shape[0])
        accuracy = 100. * correct / dna_valid.shape[0]
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%), AUPRC:{:.4f}, AUROC:{:.4f}\n'.format(
            val_loss / dna_valid.shape[0], correct, dna_valid.shape[0], accuracy, auprc, auroc))
        val_acc.append(accuracy)

        if save_name != '':
            if not os.path.exists("../data/{}".format(save_name)):
                os.mkdir("../data/{}".format(save_name))
            torch.save(model,"../data/{}/{}_epoch{}.pth".format(save_name,save_name,epoch+1))


def eval_set(val_name,tab_model = "saint_fn.pth",model_name = "ModVAR.pth",output = None,seed = 15,neg_file = "val_COSMIC_neg_0",sample = True):
    features_val,info_df = load_eval_set(val_name,neg_file,sample)
    if "23t" in val_name and sample:
        labels = features_val[6]
        pos_idx = [i for i, label in enumerate(labels) if label == 1]
        neg_idx = [i for i, label in enumerate(labels) if label == 0]
        sample_idx = []
        for i in range(3):
            sample_idx.append([pos_idx[i] for i in range(i,len(pos_idx),3)]+neg_idx)
        acc_list,prc_list,roc_list,f1_list,beat_thre_list,p_valus_list = [],[],[],[],[],[]
        for s_idx in sample_idx:
            features_val_, info_df_ = shuffle_data(0, features_val, info_df, s_idx)
            acc, prc, roc, f1, best_thre,p_value = evaluation(features_val_, info_df_, tab_model, model_name, output, seed)
            acc_list.append(acc)
            prc_list.append(prc)
            roc_list.append(roc)
            f1_list.append(f1)
            beat_thre_list.append(best_thre)
            p_valus_list.append(p_value)
        acc_np,prc_np,roc_np,f1_np,beat_thre_np,p_valus_np = np.array(acc_list),np.array(prc_list),np.array(roc_list),np.array(f1_list),np.array(beat_thre_list),np.array(p_valus_list)
        print(f"Sample result of {val_name}:")
        # print("ACC:{}±{}".format(acc_np.mean(axis=0), np.std(acc_np)))
        print("ROC:{}±{}".format(roc_np.mean(axis=0), np.std(roc_np)))
        print("PRC:{}±{}".format(prc_np.mean(axis=0), np.std(prc_np)))
        print("F1:{}±{}".format(f1_np.mean(axis=0), np.std(f1_np)))
        # print("Best thre mean:",beat_thre_np.mean(axis=0))
        # print("p_value mean:{}±{}".format(p_valus_np.mean(axis=0), np.std(p_valus_np)))

        yield acc_np,roc_np,prc_np,f1_np,beat_thre_np,p_valus_np
    else:
        yield evaluation(features_val,info_df,tab_model,model_name,output,seed)


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description='ModVAR Task')
    parser.add_argument("-m", dest='mode', default="eval", help="mode: train_eval, eval")
    parser.add_argument("-t", dest='train_file', default="train",help="train file name")
    parser.add_argument("-v", dest='val_file', default="val_23t1", help="validation file name:val_23t1,val_23t2,val_23t3,val_mutation_cluster,val_In_vivo,val_In_vitro")
    parser.add_argument('-e', dest='epoch', type=int, default=20, help="train epoch")
    parser.add_argument("-s", dest='seed', type=int, default=15, help="seed")
    parser.add_argument("-b", dest='batch_size', type=int, default=64, help="batchsize")
    parser.add_argument("-model", dest='eval_model', default="ModVAR.pth", help="full model path")
    parser.add_argument("-tab", dest='tab_model', default="saint_fn.pth", help="tabular pretrain model")
    parser.add_argument("-file", dest='file', default="val_23t1", help="file name to be processed")
    parser.add_argument("-nosample", dest='nosample', default=False, help="Use multi sampling evaluate.")
    parser.add_argument("-o", dest='output', default=None, help="if set then save the results to output directory.")
    parser.add_argument("-tool", dest='tool_name', default=None, help="Compared tool names,use ',' to separate different tools.")

    args = parser.parse_args()
    if args.nosample is not False:
        sample = False
    else:
        sample = True

    if args.mode == 'eval':
        print("Evaluation")
        if args.val_file == "all":
            val_list = ["val_23t1","val_23t2","val_23t3","val_mutation_cluster","val_In_vitro","val_In_vivo","val_disprot","civic"]
            for val_file in val_list:
                next(eval_set(val_file, tab_model=args.tab_model, model_name=args.eval_model, output=args.output,
                         seed=args.seed,sample=sample))
                print("-"*200)
        elif "civic" in args.val_file:
            neg_list = [f"val_COSMIC_neg_{i}" for i in range(10)]
            acc_list,prc_list,roc_list,f1_list = [],[],[],[]
            beat_thre_list,p_value_list = [],[]
            if args.output is not None:
                out_file = [f"{args.output}_{i}" for i in range(10)]
            else:
                out_file = [None for i in range(10)]
            z = 0
            for neg_file in neg_list:
                acc,prc,roc,f1,beat_thre,p_value = next(eval_set(args.val_file, tab_model=args.tab_model, model_name=args.eval_model, output=out_file[z],
                         seed=args.seed,neg_file=neg_file,sample=sample))
                acc_list.append(acc)
                prc_list.append(prc)
                roc_list.append(roc)
                f1_list.append(f1)
                beat_thre_list.append(beat_thre)
                p_value_list.append(p_value)
                z+=1
            acc_np,prc_np,roc_np,f1_np,beat_thre_np,p_value_np = np.array(acc_list),np.array(prc_list),np.array(roc_list),np.array(f1_list),np.array(beat_thre_list),np.array(p_value_list)
            print(f"Final result of {args.eval_model}:")
            print("ROC:{}±{}".format(roc_np.mean(axis=0), np.std(roc_np)))
            print("PRC:{}±{}".format(prc_np.mean(axis=0), np.std(prc_np)))
            print("F1:{}±{}".format(f1_np.mean(axis=0), np.std(f1_np)))
            # print("Best thre mean:", beat_thre_np.mean(axis=0))
            # print("P_value mean:", p_value_np.mean(axis=0))
        else:
            next(eval_set(args.val_file,tab_model=args.tab_model,model_name=args.eval_model,output=args.output,seed=args.seed,sample=sample))
    if args.mode == 'train_eval':
        print("Train and Eval")
        train_and_eval_ModVAR(args.train_file,
                              args.val_file,
                              args.tab_model,
                              batch_size=args.batch_size,
                              n_epoch=args.epoch,
                              save_name=args.eval_model,
                              seed=args.seed)
    if args.mode == 'extract_dna_fea':
        print("extract_dna_fea")
        fea = encode_dna_seq(args.file,save=True)
    if args.mode == 'extract_aa_fea':
        print("extract_aa_fea")
        fea = encode_aa_seq(args.file,save=True)
    if args.mode == 'calculate_metrics':
        print("calculate_metrics")
        tool_names = args.tool_name
        if tool_names == "all":
            tool_list = ["CADD_raw_rankscore","DANN_rankscore","PrimateAI_rankscore","MutationAssessor_rankscore",
                         "MVP_rankscore","gMVP_rankscore","VARITY_R_rankscore","REVEL_rankscore","M-CAP_rankscore",
                         "ESM1b_rankscore","EVE_rankscore","AlphaMissense_rankscore","CanDrA_rankscore",
                         "cscape_rankscore","fathmm_xf_coding_rankscore","css_rankscore"]
            for tool in tool_list:
                calculate_metrics(tool,args.val_file,sample)
        else:
            print(tool_names)
            for tool in tool_names.split(","):
                calculate_metrics(tool,args.val_file,sample)
    if args.mode == 'tab_pretrain':
        print("tab_pretrain")
        tab_model_pretrain(file_name=args.train_file,
                           save_name=args.tab_model,
                           num_epoch=args.epoch,
                           batch_size=args.batch_size)


if __name__ == '__main__':
    main()

'''
python task.py -m eval -v val_civic
python task.py -m train_eval -v val_mutation_cluster -model ModVAR_default
python task.py -m calculate_metrics -v val_mutation_cluster -tool all
python task.py -m extract_dna_fea -file val_mutation_cluster
python task.py -m extract_aa_fea -file val_mutation_cluster
python task.py -m tab_pretrain -file train -tab tab_default -e 500

'''