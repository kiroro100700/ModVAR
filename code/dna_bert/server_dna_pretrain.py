import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import logging

import warnings

tokenizer = AutoTokenizer.from_pretrained("dna_bert", trust_remote_code=True)
dna_model = AutoModel.from_pretrained("dna_bert", trust_remote_code=True)

def concat_delta_dna_feature(ref_list,alt_list,batch_size = 128,separate = False,mean = True,dna_model = dna_model,tokenizer=tokenizer,cuda = True):
    concat_list = []
    if separate:
        assert len(ref_list) == len(alt_list)
        for i in range(0,len(ref_list),batch_size):
            print("\r{}".format(i),end = "")
            if i+batch_size > len(ref_list):
                ref_fea = get_dna_feature(ref_list[i:],mean)
                alt_fea = get_dna_feature(alt_list[i:],mean)
            else:
                ref_fea = get_dna_feature(ref_list[i:i+batch_size],mean)
                alt_fea = get_dna_feature(alt_list[i:i+batch_size],mean)
            for j in range(ref_fea.shape[0]):
                concat_fea = np.concatenate([ref_fea[j,:], alt_fea[j,:]], axis=0)
                # print(concat_fea.shape)
                concat_list.append(concat_fea)
    else:
        ref_fea_list = get_dna_feature(ref_list)
        alt_fea_list = get_dna_feature(alt_list)
        assert ref_fea_list.shape[0] == alt_fea_list.shape[0]
        concat_list = np.concatenate((ref_fea_list,alt_fea_list),axis = 1)
    return concat_list

def get_dna_feature(dna_seq,mean = True,frozen = True,dna_model = dna_model,tokenizer =tokenizer,cuda = True):
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


use_bert_feature = True
load_bert_feature = True

seq_len = 500
ref_seqs = []
alt_seqs = []
line = ""
maxlen = 500

father_path = "dna_data"
file_name = "COSMIC_0_seq_500bp.txt"

f = open(r"{}/{}".format(father_path, file_name), "r", encoding="utf-8")

data = f.readlines()
print(len(data))
for idx in range(len(data)):
    line = data[idx].strip()
    ref_seqs.append(line.split("\t")[0])
    alt_seqs.append(line.split("\t")[1])
f.close()

print(len(ref_seqs))
print(len(alt_seqs))

z = 0
block = 10
start = z * block
for i in range(start,len(ref_seqs),block):
    len_limit = 1000
    if i + block >= len(ref_seqs):
        tmp_ref = ref_seqs[i::]
        tmp_alt = alt_seqs[i::]
    else:
        tmp_ref = ref_seqs[i : i+block]
        tmp_alt = alt_seqs[i : i+block]
    print(len(tmp_ref),len(tmp_alt))

    for idx in range(len(tmp_ref)):
        if len(tmp_ref[idx]) > len_limit:
            tmp_ref[idx] = tmp_ref[idx][0:len_limit]
    for idx in range(len(tmp_alt)):
        if len(tmp_alt[idx]) > len_limit:
            tmp_alt[idx] = tmp_alt[idx][0:len_limit]

    dataset_ori = concat_delta_dna_feature(tmp_ref, tmp_alt, batch_size=64, separate=True, mean=True)
    print(len(dataset_ori))

    dataset_ori = np.array(dataset_ori)
    np.save("output/{}_coding_set_seq_{}bp_bertfea_{}.npy".format(file_name,seq_len,z),dataset_ori)
    z += 1
    if z == 3:
        break


