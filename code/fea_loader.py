import pandas as pd
import numpy as np

def load_tool_score(file_name,mode):
    tool_df = pd.read_csv("../data/fea/tool/{}_tool.tsv".format(file_name), sep="\t", header=0)
    if mode == "infer":
        tool_df = tool_df.iloc[:,4:8]
    elif mode == "compare":
        tool_df = tool_df.iloc[:,8:]
    tool_mat = tool_df.to_numpy()
    return tool_mat

def load_labels(file_name):
    labels = []
    with open("../data/fea/dna/{}_dna_seq.txt".format(file_name),"r") as f:
        lines = f.readlines()
        for line in lines:
            labels.append(int(line.split("\t")[2]))
    return labels

def load_dna_fea(file_name):
    dna_fea = np.load("../data/fea/dna/{}_dna_fea.npy".format(file_name))
    return dna_fea

def load_aa_fea(file_name):
    if file_name == "train":
        aa_fea_1 = np.load("../data/fea/aa/{}_aa_fea_1.npy".format(file_name))
        aa_fea_2 = np.load("../data/fea/aa/{}_aa_fea_2.npy".format(file_name))
        aa_fea = np.concatenate([aa_fea_1,aa_fea_2],axis = 0)
    else:
        aa_fea = np.load("../data/fea/aa/{}_aa_fea.npy".format(file_name))
    return aa_fea

def load_gene_fea(file_name):
    gene_fea = np.load("../data/fea/gene/{}_gene_fea.npy".format(file_name))
    return gene_fea

def load_epi_fea(file_name):
    epi_fea = np.load("../data/fea/epi/{}_epi_fea.npy".format(file_name))
    return epi_fea

def load_pancan_fea(file_name):
    pancan_fea = np.load("../data/fea/pancan/{}_pancan_fea.npy".format(file_name))
    return pancan_fea

def load_fea(file_name,mode):
    labels = load_labels(file_name)
    dna_fea = load_dna_fea(file_name)
    aa_fea = load_aa_fea(file_name)
    gene_fea = load_gene_fea(file_name)
    epi_fea = load_epi_fea(file_name)
    pc_fea = load_pancan_fea(file_name)
    tool_fea = load_tool_score(file_name,mode)
    if dna_fea.shape[0] == aa_fea.shape[0] == gene_fea.shape[0] == epi_fea.shape[0] == pc_fea.shape[0] == tool_fea.shape[0] == len(labels):
        print("{} features load successfully!".format(file_name))
    else:
        raise ValueError("Feature load error!Shape not match!")

    return (dna_fea,aa_fea,gene_fea,epi_fea,pc_fea,tool_fea,labels)
