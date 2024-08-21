import fea_loader
from model import ModVAR
from data_loader import MyDataset,shuffle_data,load_data_info
from fea_extract import encode_dna_seq
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from sklearn.metrics import precision_recall_curve,roc_curve,auc
import sys
import argparse
import os

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

def train_and_eval_ModVAR(train_name,val_name,tab_model = "saint_fn",batch_size = 32,n_epoch=20,save_name=''):

    features_train = list(fea_loader.load_fea(train_name,"infer"))
    features_train_1 = []
    features_train_2 = []
    for feature in features_train:
        features_train_1.append(feature[0:4246])
        features_train_2.append(feature[4246:])
    permut_features_train_1,_ = shuffle_data(3, features_train_1)
    permut_features_train_2,_ = shuffle_data(3, features_train_2)
    features_train = []
    for i in range(len(features_train_1)):
        features_train.append(np.concatenate((permut_features_train_1[i],permut_features_train_2[i]),axis = 0))

    # features_train,_ = shuffle_data(3,features_train)
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
    features_val,_ = shuffle_data(3, features_val)

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

    set_seed(15)

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

        print('Epoch: {}, Loss: {:.5f}, Training set accuracy: {}/{} ({:.3f}%)'.format(epoch + 1, loss.item(), correct,
                                                                                       dna_train.shape[0], accuracy))
        train_acc.append(accuracy)

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

def eval_set(val_name,tab_model = "saint_fn.pth",model_name = "ModVAR.pth",output = None):
    features_val = list(fea_loader.load_fea(val_name, "infer"))
    info_df = load_data_info(val_name)
    features_val,info_df = shuffle_data(3,features_val,info_df)

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
    set_seed(15)
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

    # accuracy,Loss
    accuracy = 100. * correct / dna_valid.shape[0]
    print('Test set: Accuracy: {}/{} ({:.3f}%), AUPRC:{:.4f}, AUROC:{:.4f}\n'.format(
         correct, dna_valid.shape[0], accuracy, auprc, auroc))
    if output is not None:
        print("Writing the results into output...")
        with open("../output/{}.tsv".format(output),"w+") as f:
            f.write("Chr\tPos\tRef\tAlt\tLabel\tScore\n")
            for i in range(len(label_all)):
                line = '{}\t{}\t{}\t{}\t{}\t{:.4f}\n'.format(info_df.iloc[i,0],info_df.iloc[i,1],info_df.iloc[i,2],info_df.iloc[i,3],label_all[i],prob_all[i])
                f.write(line)
        print("Successfully write into {}.".format(output))

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
    parser.add_argument("-o", dest='output', default=None, help="if set then save the results to output directory.")

    args = parser.parse_args()

    if args.mode == 'eval':
        print("Evaluation")
        eval_set(args.val_file,tab_model=args.tab_model,model_name=args.eval_model,output=args.output)
    if args.mode == 'train_eval':
        print("Train and Eval")
        train_and_eval_ModVAR(args.train_file,
                              args.val_file,
                              args.tab_model,
                              batch_size=args.batch_size,
                              n_epoch=args.epoch,
                              save_name=args.eval_model)
    if args.mode == 'extract_dna_fea':
        print("extract_dna_fea")
        fea = encode_dna_seq(args.file,save=True)



if __name__ == '__main__':
    main()

'''
python task.py -m eval -v val_23t1 -o test
python task.py -m train_eval -v val_23t1 -model ModVAR_default
python task.py -m extract_dna_fea -file val_In_vivo
'''