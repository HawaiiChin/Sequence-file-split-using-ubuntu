import os
import sys
import random
import math
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_auc_score)
RANDOM=42
random.seed(RANDOM)

np.random.seed(RANDOM)
def read_fasta(fasta_path):
    sequences={}
    seq_id=None
    seq_chunks = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id is not None:
                    sequences[seq_id]="".join(seq_chunks)
                seq_id=line[1:]
                seq_chunks=[]
            else:
                seq_chunks.append(line.upper())

        if seq_id is not None:
            sequences[seq_id]="".join(seq_chunks)
    return sequences

def write_fasta(sequences, output_path):
    with open(output_path, "w") as f:
        for seq_id, seq in sequences.items():
            f.write(f">{seq_id}\n")
            f.write(f"{seq}\n")


def random_dna(length):
    return "".join(random.choices(["A", "T", "G", "C"], k = length))

def simulate_dataset(n_sequences=200, min_len=200, max_len=500):
    sequences={}
    labels={}
    for i in range(n_sequences):
        length=random.randint(min_len, max_len)
        seq=random_dna(length)
        gc_content=(seq.count("G")+seq.count("C"))/len(seq)
        label=1 if gc_content> 0.52 else 0
        seq_id= f"seq_{i}"
        sequences[seq_id]=seq
        labels[seq_id]=label

    return sequences, labels

def get_kmers(sequence,k=4):
    return [sequence[i:i+4] 
            for i in range(len(sequence)-k+1)]

def build_kmer_vocabulary(sequences, k=4):
    vocab=set()
    for seq in sequences.values():
        vocab.update(get_kmers(seq, k))
    return sorted(vocab)

def kmer_count_vector(sequence, vocab, k=4):
    counts=Counter(get_kmers(sequence, k))
    return np.array([counts.get(kmer,0) for kmer in vocab])

def build_feature_matrix(sequences, vocab, k=4):
    X=[]
    for seq in sequences.values():
        X.append(kmer_count_vector(seq,vocab,k))
    return np.vstack(X)

def plot_gc_distribution(sequences,labels):
    gc_0, gc_1=[],[]
    for seq_id, seq in sequences.items():
        gc=(seq.count("G")+seq.count("C"))/len(seq)
        if labes[seq_id]==0:
            gc_0.append(gc)
        else:
            gc_1.append(gc)
    plt.hist(gc_0, bins=20,alpha=0.6,label="Class 0")
    plt.hist(gc_1,bins=20, alpha=0.6, label="Class 1")
    plt.savefig("fig.png"
                )
