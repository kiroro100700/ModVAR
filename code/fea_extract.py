import numpy as np
from transformers import AutoTokenizer, AutoModel
from transformers import logging
import warnings
import torch
import esm
import numpy as np

def load_dna_bert():
    tokenizer = AutoTokenizer.from_pretrained("dna_bert", trust_remote_code=True)
    dna_model = AutoModel.from_pretrained("dna_bert", trust_remote_code=True)
    return tokenizer,dna_model

def load_esm():
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()  # 1280
    # esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D() #480
    batch_converter = alphabet.get_batch_converter()
    return esm_model,batch_converter

def get_dna_feature(dna_seq, dna_model,tokenizer,mean = True,frozen = True,cuda = False):
    logging.set_verbosity_error()
    warnings.filterwarnings("ignore")
    inputs = tokenizer(dna_seq, return_tensors='pt',padding = True,truncation=True)["input_ids"]
    # print(inputs.shape)
    if cuda:
        inputs = inputs.cuda()
    hidden_states = dna_model(inputs)[0]  # [1, sequence_length, 768]
    if mean:
        # embedding with mean pooling
        if hidden_states.dim()>2:
            hidden_states = torch.mean(hidden_states, dim=1)
        else:
            hidden_states = torch.mean(hidden_states, dim=0)
        # print(embedding_mean.shape)  # expect to be 768
    if frozen:
        return hidden_states.detach().numpy()
    else:
        return hidden_states


def get_aa_feature(data,model,device,batch_converter,aa_len=1280,batch_size = 2):
    np.random.seed(0)
    max_len = 500
    token_representations = []
    for i in range(0, len(data), batch_size):
        print(i)
        if i + batch_size >= len(data):
            b_data = data[i:]
        else:
            b_data = data[i:i + batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(b_data)

        with torch.no_grad():
            print(batch_tokens.shape[1])
            # split the seq into small blocks when too long
            if batch_tokens.shape[1] > max_len:
                res_tmp = np.zeros((2, aa_len))
                start = True
                end = False
                for j in range(batch_tokens.shape[1] // max_len + 1):
                    if j == batch_tokens.shape[1] // max_len:
                        print(j * max_len, "-- end")
                        end = True
                        tokens = batch_tokens[:, j * max_len:]
                    else:
                        print(j * max_len, "--", (j + 1) * max_len)
                        tokens = batch_tokens[:, j * max_len:(j + 1) * max_len]
                    tokens = tokens.to(device)
                    results = model(tokens, repr_layers=[33], return_contacts=True)
                    token_res = results["representations"][33].detach().cpu().numpy()
                    if start:
                        start = False
                        print(token_res[1][1:].shape)
                        res_tmp[0] = np.sum(token_res[0][1:], axis=0) / batch_tokens.shape[1] + res_tmp[0]
                        res_tmp[1] = np.sum(token_res[1][1:], axis=0) / batch_tokens.shape[1] + res_tmp[1]
                    elif end:
                        end = False
                        print(token_res[1][:len(token_res[0]) - 1].shape)
                        res_tmp[0] = np.sum(token_res[0][:len(token_res[0]) - 1], axis=0) / batch_tokens.shape[1] + \
                                     res_tmp[0]
                        res_tmp[1] = np.sum(token_res[1][:len(token_res[1]) - 1], axis=0) / batch_tokens.shape[1] + \
                                     res_tmp[1]
                    else:
                        print(token_res[1].shape)
                        res_tmp[0] = np.sum(token_res[0], axis=0) / batch_tokens.shape[1] + res_tmp[0]
                        res_tmp[1] = np.sum(token_res[1], axis=0) / batch_tokens.shape[1] + res_tmp[1]
            else:
                batch_tokens = batch_tokens.to(device)
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                token_res = results["representations"][33].detach().cpu().numpy()

                res_tmp = np.zeros((2, aa_len))
                res_tmp[0] = token_res[0][1: len(token_res[0]) - 1].mean(0)
                res_tmp[1] = token_res[1][1: len(token_res[1]) - 1].mean(0)

            token_representations.append(res_tmp)
    return np.array(token_representations)

def concat_delta_dna_feature(ref_list,alt_list,batch_size = 32,separate = False,mean = True,cuda = False):
    tokenizer, dna_model = load_dna_bert()
    concat_list = []
    if separate:
        assert len(ref_list) == len(alt_list)
        for i in range(0,len(ref_list),batch_size):
            print("\r{}".format(i),end = "")
            if i+batch_size > len(ref_list):
                ref_fea = get_dna_feature(ref_list[i:],dna_model,tokenizer,mean)
                alt_fea = get_dna_feature(alt_list[i:],dna_model,tokenizer,mean)
            else:
                ref_fea = get_dna_feature(ref_list[i:i+batch_size],dna_model,tokenizer,mean)
                alt_fea = get_dna_feature(alt_list[i:i+batch_size],dna_model,tokenizer,mean)
            for j in range(ref_fea.shape[0]):
                concat_fea = np.concatenate([ref_fea[j,:], alt_fea[j,:]], axis=0)
                # print(concat_fea.shape)
                concat_list.append(concat_fea)
    else:
        ref_fea_list = get_dna_feature(ref_list,dna_model,tokenizer)
        alt_fea_list = get_dna_feature(alt_list,dna_model,tokenizer)
        assert ref_fea_list.shape[0] == alt_fea_list.shape[0]
        concat_list = np.concatenate((ref_fea_list,alt_fea_list),axis = 1)
    return concat_list

def concat_delta_aa_feature(aa_mat,aa_len = 1280):
    print(aa_mat.shape)
    aa_mat = np.nan_to_num(aa_mat)
    aa_feas = np.zeros((aa_mat.shape[0], aa_len * 2))
    i = 0
    for l in aa_mat:
        ref = l[0]
        alt = l[1]
        cat = np.concatenate([ref, alt], axis=0)
        aa_feas[i] = cat
        i += 1
    print(aa_feas.shape)

    return aa_feas

def encode_dna_seq(file_name,save = False):
    len_limit = 1000
    ref_seqs = []
    alt_seqs = []
    with open("../data/fea/dna/{}_dna_seq.txt".format(file_name),"r",encoding = "utf-8") as f:
        for line in f.readlines():
            ref_seqs.append(line.split("\t")[0])
            alt_seqs.append(line.split("\t")[1])

    for idx in range(len(ref_seqs)):
        if len(ref_seqs[idx]) > len_limit:
            ref_seqs[idx] = ref_seqs[idx][0:len_limit]
    for idx in range(len(alt_seqs)):
        if len(alt_seqs[idx]) > len_limit:
            alt_seqs[idx] = alt_seqs[idx][0:len_limit]

    dna_fea = concat_delta_dna_feature(ref_seqs, alt_seqs, batch_size=64, separate=True, mean=True)
    if save:
        np.save("../data/fea/dna/{}_dna_fea.npy".format(file_name), dna_fea)
    print("fea shape:"+str(np.array(dna_fea).shape))
    print("extract successfully!")
    return dna_fea


def encode_aa_seq(file_name,save = False):
    data = []
    f = open('../data/fea/aa/{}_aa_seq.txt'.format(file_name),"r",encoding="utf-8")
    strings = f.readlines()
    i = 0
    for line in strings:
        ref_tag = "ref_{}".format(i)
        ref_seq = line.split("\t")[0]
        tup = (ref_tag,ref_seq)
        data.append(tup)

        alt_tag = "alt_{}".format(i)
        alt_seq = line.split("\t")[1]
        tup = (alt_tag,alt_seq)
        data.append(tup)
        i+=1
    f.close()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    esm_model,batch_converter = load_esm()
    esm_model.to(device)
    esm_model.eval()
    aa_mat = get_aa_feature(data,esm_model,device,batch_converter)
    aa_feas = concat_delta_aa_feature(aa_mat)
    if save:
        np.save("../data/fea/aa/{}_aa_fea.npy".format(file_name), aa_feas)
    print("fea shape:"+str(aa_feas.shape))
    print("extract successfully!")
    return aa_feas
