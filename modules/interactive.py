import numpy as np
import torch
import json

from .tokenizers import Tokenizer

def idx2token(ids):
    with open("data/vocab/idx2token.json", "r") as fp2:
        idx2token_dict = json.load(fp2)

    txt = ''
    
    if type(ids) == int:
        txt = idx2token_dict[str(ids)]
        return txt
        
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

class Interactive(object):
    def __init__(self, mode, length=None, threshold=None):
        self.length = length
        self.mode = mode
        self.threshold = threshold
        
    def sentence_base(self, tgt):
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

    def length_base(self, tgt):
        if len(tgt[0]) % self.length == 1 and len(tgt[0]) > 1:
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
    
    def interactive_tgt(self, tgt):
        if self.mode == 'sentence':
            tgt = self.sentence_base(tgt)
        if self.mode == 'length':
            tgt == self.length_base(tgt)
        return tgt
    
    # def interactive_state(self, it, sampleLogprobs, state):
    #     if self.mode == 'confidence':
    
    def confidence_base(self, it, sampleLogprobs, state):
        next_prob = float(torch.exp(sampleLogprobs))
        
        if next_prob < self.threshold:
            print('sentence have been generated:', idx2token(state[0][0][0].numpy()))
            print("next token is:", idx2token(int(it)), '\t', 'token probability is:', next_prob)
            new_token = input("input your new token: ").lower()
            if len(new_token) == 0:
                pass
            else:
                it = torch.tensor([token2idx(new_token)[0]])
                sampleLogprobs = torch.tensor(np.log(np.array(self.threshold)))
        return it, sampleLogprobs