import numpy as np
import torch
import json

def idx2token(ids):
    with open("data/vocab/idx2token.json", "r") as fp2:
        idx2token_dict = json.load(fp2)

    txt = ''
    for i, idx in enumerate(ids):
        if idx == 0 and i == 0:
            pass
        elif idx == 1:
            txt = txt.rstrip()
            txt += idx2token_dict[str(idx)]
            txt += ' '
        elif idx > 0:
            txt += idx2token_dict[str(idx)]
            txt += ' '
    return txt

def token2idx(string):
    with open("data/vocab/token2idx.json", "r") as fp1:
        token2idx_dict = json.load(fp1)

    ids = []
    tokens = string.replace(".", " .").split()
    for token in tokens:
        if token != ' ':
            ids.append(token2idx_dict[token])
    return ids

def sentence_base(tgt):
    if tgt[0][-1] == 1:
        ids = tgt.numpy()
        print("sentence you can edit:", idx2token(ids[0]))
        new_string = input("input your new string: ").lower()
        if len(new_string) == 0:
            pass
        else:
            new_ids = token2idx(new_string)
            while len(ids[0]) > len(new_ids):
                new_ids.insert(0, 0)
                ids[0] = new_ids
                tgt = torch.from_numpy(ids)
    return tgt

def length_base(tgt, length):
    if len(tgt[0]) % length == 1 and len(tgt[0]) > 1:
        ids = tgt.numpy()
        print("sentence you can edit:", idx2token(ids[0]))
        new_string = input("input your new string: ").lower()
        if len(new_string) == 0:
            pass
        else:
            new_ids = token2idx(new_string)
            while len(ids[0]) > len(new_ids):
                new_ids.insert(0, 0)
                ids[0] = new_ids
                tgt = torch.from_numpy(ids)
    return tgt