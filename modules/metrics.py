from nltk.translate.bleu_score import sentence_bleu
import pickle
import json
from tqdm import tqdm
import evaluate
from datasets import load_metric

def compute_others(gts, res, mode, threshold):
    
    meteor = evaluate.load('meteor')
    rouge = evaluate.load('rouge')
    scores = {'meteor': 0,
             'rouge-l': 0}
    if mode == 'sentence':
        for i in range(len(res)):
            idx = len(gts[i].split('.')[0])
            gt_process = gts[i][idx+1: ].split()
            res_process = res[i][idx+1: ].split()
            scores['meteor'] += meteor.compute(predictions=[' '.join(res_process)], references=[' '.join(gt_process)])['meteor']/len(res)
            scores['rouge-l'] += rouge.compute(predictions=[' '.join(res_process)], references=[' '.join(gt_process)])['rougeL']/len(res)
            
    elif mode == 'length':
        for i in range(len(res)):
            idx = threshold - 1
            gt_process = gts[i][idx+1: ].split()
            res_process = res[i][idx+1: ].split()
            scores['meteor'] += meteor.compute(predictions=[' '.join(res_process)], references=[' '.join(gt_process)])['meteor']/len(res)
            scores['rouge-l'] += rouge.compute(predictions=[' '.join(res_process)], references=[' '.join(gt_process)])['rougeL']/len(res)
            
    else:
        for i in range(len(res)):
           scores['meteor'] += meteor.compute(predictions=[res[i]], references=[gts[i]])['meteor']/len(res)
           scores['rouge-l'] += rouge.compute(predictions=[res[i]], references=[gts[i]])['rougeL']/len(res)
        
    return scores

def compute_bleu(gts, res, mode, threshold):
    bleu_scores = {'BLEU 1': 0,
                    'BLEU 2': 0,
                    'BLEU 3': 0,
                    'BLEU 4': 0,
                    'BLEU S': 0}
    if mode == 'sentence':
        for i in range(len(res)):
            idx = len(gts[i].split('.')[0])
            gt_process = gts[i][idx+1: ].split()
            res_process = res[i][idx+1: ].split()
            bleu_scores['BLEU 1'] += sentence_bleu([gt_process], res_process, weights=(1,0,0,0))/len(res)
            bleu_scores['BLEU 2'] += sentence_bleu([gt_process], res_process, weights=(0,1,0,0))/len(res)
            bleu_scores['BLEU 3'] += sentence_bleu([gt_process], res_process, weights=(0,0,1,0))/len(res)
            bleu_scores['BLEU 4'] += sentence_bleu([gt_process], res_process, weights=(0,0,0,1))/len(res)
            bleu_scores['BLEU S'] += sentence_bleu([gt_process], res_process)/len(res)
            
    elif mode == 'length':
        for i in range(len(res)):
            idx = threshold - 1
            gt_process = gts[i][idx+1: ].split()
            res_process = res[i][idx+1: ].split()
            bleu_scores['BLEU 1'] += sentence_bleu([gt_process], res_process, weights=(1,0,0,0))/len(res)
            bleu_scores['BLEU 2'] += sentence_bleu([gt_process], res_process, weights=(0,1,0,0))/len(res)
            bleu_scores['BLEU 3'] += sentence_bleu([gt_process], res_process, weights=(0,0,1,0))/len(res)
            bleu_scores['BLEU 4'] += sentence_bleu([gt_process], res_process, weights=(0,0,0,1))/len(res)
            bleu_scores['BLEU S'] += sentence_bleu([gt_process], res_process)/len(res)
            
    else:
        for i in range(len(res)):
            bleu_scores['BLEU 1'] += sentence_bleu([gts[i].split()], res[i].split(), weights=(1,0,0,0))/len(res)
            bleu_scores['BLEU 2'] += sentence_bleu([gts[i].split()], res[i].split(), weights=(0,1,0,0))/len(res)
            bleu_scores['BLEU 3'] += sentence_bleu([gts[i].split()], res[i].split(), weights=(0,0,1,0))/len(res)
            bleu_scores['BLEU 4'] += sentence_bleu([gts[i].split()], res[i].split(), weights=(0,0,0,1))/len(res)
            bleu_scores['BLEU S'] += sentence_bleu([gts[i].split()], res[i].split())/len(res)
            
    return bleu_scores


def compute_scores(gts, res, mode, threshold):
    scores = {}
    bleu_scores = compute_bleu(gts, res, mode, threshold)
    scores.update(bleu_scores)
    other_scores = compute_others(gts, res, mode, threshold)
    scores.update(other_scores)
    return scores