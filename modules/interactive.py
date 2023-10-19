import numpy as np
import torch
import json
from tkinter import *
from tkinter import simpledialog

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

def get_args():
    window = Tk()
    window.geometry("500x300")
    window.title("Define Arguments")

    threshold = None

    # this method accepts integer and returns integer
    mode = simpledialog.askstring("Mode","Enter interaction mode",parent=window)
    if mode == 'confidence':
        threshold = simpledialog.askfloat('Threshold', 'Set your confidence threshold', parent=window)
        dialog_output = Label(window, text=f'Interaction mode is {mode}, confidence threshold is {threshold}.',font=('italic 12'))
    elif mode == 'length':
        threshold = simpledialog.askinteger('Length', 'Set your length threshold', parent=window)
        dialog_output = Label(window, text=f'Interaction mode is {mode}, length threshold is {threshold}.',font=('italic 12'))
    else:
        dialog_output = Label(window, text=f'Interaction mode is {mode}.',font=('italic 12'))
    dialog_output.pack(pady=20)
    
    print(mode, threshold)

    # window.mainloop()
    
    return mode, threshold

def window(str1, str2):
    window = Tk()
    window.geometry("500x300")
    window.title('Interactive Generation')
    new_string = simpledialog.askstring(str1, str2, parent=window)
    dialog_output = Label(window, text=f'Edited generation is {new_string}.',font=('italic 12'))
        
    return new_string
    
class Interactive(object):
    def __init__(self, mode, threshold):
        self.mode = mode
        self.threshold = threshold
        
    def sentence_base(self, it, state, auto_eval, targets, flag_edit):
        curr_it = it[0]
        if curr_it == 1:
            if auto_eval == False:
                ids = state[0].reshape(-1).numpy()
                str1 = 'Sentence-based Interaction'
                str2 = 'Sentence you can edit: ' + idx2token(ids) + '\n\nEnter your new sentence: '
                new_string = window(str1, str2).lower()
                if len(new_string) == 0:
                    pass
                else:
                    new_ids = token2idx(new_string)
                    it = torch.tensor([1])
                    state = [torch.tensor([[new_ids]])]
            elif auto_eval == True:
                if flag_edit == False:
                    ids = targets[0].tolist()
                    end_idx = ids.index(1)
                    new_ids = ids[:end_idx]
                    it = torch.tensor([1]) # here is "."
                    state = [torch.tensor([[new_ids]])]
                    flag_edit = True
                    # print('auto-edit', state)
                else:
                    pass
        return it, state, flag_edit
    
    def length_base(self, it, state, auto_eval, targets, flag_edit):
        curr_gen = state[0].reshape(-1) # the first token is 0
        if (len(curr_gen)-1) % self.threshold == 0 and len(curr_gen) > 1:
            if auto_eval == False:
                ids = curr_gen.numpy()
                str1 = 'Length-base Interaction'
                str2 = 'Sentence you can edit: ' + idx2token(ids) + '\n\nEnter your new string: '
                new_string = window(str1, str2).lower()
                if len(new_string) == 0:
                    pass
                else:
                    new_ids = token2idx(new_string)
                    it = torch.tensor([new_ids[-1]])
                    state = [torch.tensor([[new_ids[:-1]]])]
            elif auto_eval == True and flag_edit == False:
                ids = targets[0].tolist() # first token is 0
                new_ids = ids[:self.threshold+1]
                it = torch.tensor([ids[self.threshold]])
                state = [torch.tensor([[new_ids[:-1]]])]
                flag_edit = True
                # print('auto-edit', state)
        return it, state, flag_edit
    
    def confidence_base(self, it, sampleLogprobs, state):
        next_prob = float(torch.exp(sampleLogprobs))
        
        if next_prob < self.threshold and (next_token not in self.stop_words):
            if auto_eval == False:
                str1 = 'Confidence-based Interaction'
                str2 = 'Sentence have been generated: ' + idx2token(state[0][0][0].numpy()) + \
                    '\nNext token: ' + idx2token(int(it)) + ', Next token probability: ' + str(next_prob) + \
                    '\n\nEnter your new string: '
                new_string = window(str1, str2).lower()
                if len(new_string) == 0:
                    pass
                else:
                    new_ids = token2idx(new_string)
                    it = torch.tensor([new_ids[-1]])
                    state = [torch.tensor([[new_ids[:-1]]])]
            elif auto_eval == True and flag_edit == False:
                cur_gen = idx2token(state[0].reshape(-1).numpy())
                gt = idx2token(targets[0].tolist())
                report_dict = {'Report Impression': [cur_gen, gt]}
                report_df = pd.DataFrame(report_dict)
                report_df.to_csv('modules/labeler/temp_reports.csv')
                classifier = Classifier('modules/labeler/chexbert.pth', 'modules/labeler/temp_reports.csv')
                pred = classifier.label()
                pred = np.array(pred).T
                labels = []
                for i in pred:
                    temp_labels = []
                    for idx in range(14):
                        if i[idx] == 1:
                            temp_labels.append(str(CONDITIONS[idx])+": positive")
                        elif i[idx] == 2:
                            temp_labels.append(str(CONDITIONS[idx])+": negative")                        
                        elif i[idx] == 3:
                            temp_labels.append(str(CONDITIONS[idx])+": uncertain")
                    if len(temp_labels) == 0:
                        temp_labels.append("not assigned to any class yet")
                    labels.append(temp_labels)
                # print(labels)
                flag_edit = True
                str1 = 'Confidence-based Interaction Classifier Mode'
                str2 = 'Sentence have been generated: ' + cur_gen + \
                    '\nNext token: ' + str(next_token) + ', Next token probability: ' + str(next_prob) + \
                    '\nCategories for current generation: ' + ', '.join(labels[0]) + \
                    '\nCategories for ground truth: ' + ', '.join(labels[1]) + \
                    '\n\nEnter your new string: '
                
                new_string = window(str1, str2).lower()
                # print('Sentence have been generated: ' + cur_gen + \
                #     '\nNext token: ' + str(next_token) + ', Next token probability: ' + str(next_prob) + \
                #     '\nCategories for current generation: ' + ', '.join(labels[0]) + \
                #     '\nCategories for ground truth: ' + ', '.join(labels[1]) + \
                #     '\n\nEnter your new string: ')
                # new_string = input('your inputs: ').lower()
                if len(new_string) == 0:
                    pass
                else:
                    new_ids = token2idx(new_string)
                    it = torch.tensor([new_ids[-1]])
                    state = [torch.tensor([[new_ids[:-1]]])]
                
        return it, state, flag_edit
