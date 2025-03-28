from os.path import join

from transformers import *
import torch
import pdb

import torch
import numpy as np
import os
torch.manual_seed(37)
torch.cuda.manual_seed(37)

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
np.random.seed(37)
import torch.optim as optim

TOKEN_LEN = 50
VOCAB_SIZE = 100000
LaBSE_DIM = 256
EMBED_DIM = 300
BATCH_SIZE = 96
FASTTEXT_DIM = 300
NEIGHBOR_SIZE = 20
ATTENTION_DIM = 300
MULTI_HEAD_DIM = 1



class LaBSEEncoder(nn.Module):
    def __init__(self, DATA_DIR="Mydata"):
        super(LaBSEEncoder, self).__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(join(DATA_DIR, "LaBSE"), do_lower_case=False)
        self.model = AutoModel.from_pretrained(join(DATA_DIR, "LaBSE")).to(self.device)

    def forward(self, batch):
        sentences = batch
        #  text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) or `List[List[str]]` (batch of pretokenized examples).
        tok_res = self.tokenizer(sentences, add_special_tokens=True, padding='max_length', max_length=MAX_LEN)
        input_ids = torch.LongTensor([d[:MAX_LEN] for d in tok_res['input_ids']]).to(self.device)
        token_type_ids = torch.LongTensor(tok_res['token_type_ids']).to(self.device)
        attention_mask = torch.LongTensor(tok_res['attention_mask']).to(self.device)
        output = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return F.normalize(output[0][:, 1:-1, :].sum(dim=1))




if __name__ == "__main__":
    device = "cuda:0"


    MAX_LEN = 130


    path = "Mydata/DWY/"
    model = LaBSEEncoder().to(device)
   
    ent_labse_embed=torch.zeros(20665,768)
    rel_labse_embed=torch.zeros(2947,768)
    
    with open("Mydata/cleaned_ent_ids_1.txt", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            l = line.split('\t')
            id = int(l[0])
            ent_name = str(l[1])
            ent_labse_embed[id,:] = model([ent_name]).cpu().detach()
    with open("Mydata/cleaned_ent_ids_2.txt", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            l = line.split('\t')
            id = int(l[0])
            ent_name = str(l[1])
            ent_labse_embed[id] = model([ent_name]).cpu().detach()
    with open("Mydata/cleaned_ent_ids_3.txt", encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                l = line.split('\t')
                id = int(l[0])
                ent_name = str(l[1])
                ent_labse_embed[id] = model([ent_name]).cpu().detach()
    with open("Mydata/cleaned_ent_ids_4.txt", encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                l = line.split('\t')
                id = int(l[0])
                ent_name = str(l[1])
                ent_labse_embed[id] = model([ent_name]).cpu().detach()
    save_dict = open('Mydata/ent_LaBSE.txt', 'wb')
    pickle.dump(np.array(ent_labse_embed),save_dict)
    """rel emb"""
    with open("Mydata/cleaned_rel_ids_1.txt", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            l = line.split('\t')
            id = int(l[0])
            rel_name = str(l[1])
            rel_labse_embed[id,:] = model([rel_name]).cpu().detach()
    with open("Mydata/cleaned_rel_ids_2.txt", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            l = line.split('\t')
            id = int(l[0])
            rel_name = str(l[1])
            rel_labse_embed[id] = model([rel_name]).cpu().detach()
    with open("Mydata/cleaned_rel_ids_3.txt", encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                l = line.split('\t')
                id = int(l[0])
                rel_name = str(l[1])
                rel_labse_embed[id] = model([rel_name]).cpu().detach()
    with open("Mydata/cleaned_rel_ids_4.txt", encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                l = line.split('\t')
                id = int(l[0])
                rel_name = str(l[1])
                rel_labse_embed[id] = model([rel_name]).cpu().detach()
    save_dict = open('Mydata/rel_LaBSE.txt', 'wb')
    pickle.dump(np.array(rel_labse_embed),save_dict)

print("OK!")