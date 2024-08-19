import numpy as np
from transformers import AutoTokenizer, AutoModel
from transformers import logging
import warnings
import torch

def load_dna_bert():
    tokenizer = AutoTokenizer.from_pretrained("dna_bert", trust_remote_code=True)
    dna_model = AutoModel.from_pretrained("dna_bert", trust_remote_code=True)
    return tokenizer,dna_model

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
    return dna_fea
