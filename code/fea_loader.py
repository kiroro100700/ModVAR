import pandas as pd
import numpy as np

def load_compare_score(file_name,tool_name,benchmark_dir=False):
    if not benchmark_dir:
        father_path = "tool"
    else:
        father_path = file_name
    tool_df = pd.read_csv("../data/fea/{}/{}_tool.tsv".format(father_path, file_name), sep="\t", header=0)
    tool_df = tool_df.loc[:,tool_name]
    return tool_df


def load_tool_score(file_name,mode,benchmark_dir=False,from_np=False):
    if not benchmark_dir:
        father_path = "tool"
    else:
        father_path = file_name
    if not from_np:
        tool_df = pd.read_csv("../data/fea/{}/{}_tool.tsv".format(father_path,file_name), sep="\t", header=0)
        if mode == "infer":
            tool_df = tool_df.iloc[:,4:8]
        elif mode == "compare":
            tool_df = tool_df.iloc[:,8:]
        tool_mat = tool_df.to_numpy()
    else:
        tool_mat = np.load("../data/fea/{}/{}_tool.npy".format(father_path,file_name),allow_pickle=True)

    return tool_mat

def load_labels(file_name,benchmark_dir=False):
    if not benchmark_dir:
        father_path = "dna"
    else:
        father_path = file_name
    labels = []
    with open("../data/fea/{}/{}_dna_seq.txt".format(father_path,file_name),"r") as f:
        lines = f.readlines()
        for line in lines:
            labels.append(int(line.split("\t")[2]))
    return labels

def load_dna_fea(file_name,benchmark_dir=False):
    if not benchmark_dir:
        father_path = "dna"
    else:
        father_path = file_name
    dna_fea = np.load("../data/fea/{}/{}_dna_fea.npy".format(father_path,file_name))
    return dna_fea

def load_aa_fea(file_name,benchmark_dir=False):
    if not benchmark_dir:
        father_path = "aa"
    else:
        father_path = file_name
    if file_name == "train":
        aa_fea_1 = np.load("../data/fea/{}/{}_aa_fea_1.npy".format(father_path,file_name))
        aa_fea_2 = np.load("../data/fea/{}/{}_aa_fea_2.npy".format(father_path,file_name))
        aa_fea = np.concatenate([aa_fea_1,aa_fea_2],axis = 0)
    else:
        aa_fea = np.load("../data/fea/{}/{}_aa_fea.npy".format(father_path,file_name))
    return aa_fea

def load_gene_fea(file_name,benchmark_dir=False):
    if not benchmark_dir:
        father_path = "gene"
    else:
        father_path = file_name
    gene_fea = np.load("../data/fea/{}/{}_gene_fea.npy".format(father_path,file_name))
    return gene_fea

def load_epi_fea(file_name,benchmark_dir=False):
    if not benchmark_dir:
        father_path = "epi"
    else:
        father_path = file_name
    epi_fea = np.load("../data/fea/{}/{}_epi_fea.npy".format(father_path,file_name))
    return epi_fea

def load_pancan_fea(file_name,benchmark_dir=False):
    if not benchmark_dir:
        father_path = "pancan"
    else:
        father_path = file_name
    pancan_fea = np.load("../data/fea/{}/{}_pancan_fea.npy".format(father_path,file_name))
    return pancan_fea

def load_fea(file_name,mode,benchmark_dir=False,load_label = True):

    dna_fea = load_dna_fea(file_name,benchmark_dir)
    aa_fea = load_aa_fea(file_name,benchmark_dir)
    gene_fea = load_gene_fea(file_name,benchmark_dir)
    epi_fea = load_epi_fea(file_name,benchmark_dir)
    pc_fea = load_pancan_fea(file_name,benchmark_dir)
    tool_fea = load_tool_score(file_name,mode,benchmark_dir)
    if load_label:
        labels = load_labels(file_name,benchmark_dir)
    else:
        labels = [1 for _ in range(dna_fea.shape[0])]
    # print(dna_fea.shape,aa_fea.shape,gene_fea.shape, epi_fea.shape, pc_fea.shape, tool_fea.shape,len(labels))
    if dna_fea.shape[0] == aa_fea.shape[0] == gene_fea.shape[0] == epi_fea.shape[0] == pc_fea.shape[0] == tool_fea.shape[0]:
        print("{} features load successfully!".format(file_name))
    else:
        raise ValueError("Feature load error!Shape not match!")

    return dna_fea,aa_fea,gene_fea,epi_fea,pc_fea,tool_fea,labels

def load_COSMIC_neg_fea(file_name,mode,benchmark_dir=True):
    dna_fea = load_dna_fea(file_name,benchmark_dir)
    aa_fea = load_aa_fea(file_name,benchmark_dir)
    gene_fea = load_gene_fea(file_name,benchmark_dir)
    epi_fea = load_epi_fea(file_name,benchmark_dir)
    pc_fea = load_pancan_fea(file_name,benchmark_dir)
    tool_fea = load_tool_score(file_name,mode,benchmark_dir,from_np=True)
    labels = [0 for _ in range(dna_fea.shape[0])]
    if dna_fea.shape[0] == aa_fea.shape[0] == gene_fea.shape[0] == epi_fea.shape[0] == pc_fea.shape[0] == tool_fea.shape[0] == len(labels):
        print("{} features load successfully!".format(file_name))
    else:
        raise ValueError("Feature load error!Shape not match!")

    return (dna_fea,aa_fea,gene_fea,epi_fea,pc_fea,tool_fea,labels)

def load_tabular_fea(file_name):
    gene_fea = load_gene_fea(file_name)
    epi_fea = load_epi_fea(file_name)
    pc_fea = load_pancan_fea(file_name)
    if gene_fea.shape[0] == epi_fea.shape[0] == pc_fea.shape[0]:
        print("{} features load successfully!".format(file_name))
    else:
        raise ValueError("Feature load error!Shape not match!")

    return (gene_fea,epi_fea,pc_fea)
