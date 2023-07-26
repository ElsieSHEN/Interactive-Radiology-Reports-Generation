from nltk.translate.bleu_score import sentence_bleu
import pickle

def compute_bleu(gts, res, mode, threshold):
    bleu_scores = {'BLEU 1': 0,
                    'BLEU 2': 0,
                    'BLEU 3': 0,
                    'BLEU 4': 0,
                    'BLEU S': 0}
    if mode == 'sentence':
        for i in range(len(res)):
            idx = len(gt[i].split('.')[0])
            gt_process = gt[i][idx+1: ].split()
            res_process = res[i][idx+1: ].split()
            bleu_scores['BLEU 1'] += sentence_bleu([gt_process], res_process, weights=(1,0,0,0))/len(res)
            bleu_scores['BLEU 2'] += sentence_bleu([gt_process], res_process, weights=(0,1,0,0))/len(res)
            bleu_scores['BLEU 3'] += sentence_bleu([gt_process], res_process, weights=(0,0,1,0))/len(res)
            bleu_scores['BLEU 1'] += sentence_bleu([gt_process], res_process, weights=(0,0,0,1))/len(res)
            bleu_scores['BLEU S'] += sentence_bleu([gt_process], res_process)/len(res)
            
    if mode == 'length':
        for i in range(len(res)):
            idx = threshold - 1
            gt_process = gt[i][idx+1: ].split()
            res_process = res[i][idx+1: ].split()
            bleu_scores['BLEU 1'] += sentence_bleu([gt_process], res_process, weights=(1,0,0,0))/len(res)
            bleu_scores['BLEU 2'] += sentence_bleu([gt_process], res_process, weights=(0,1,0,0))/len(res)
            bleu_scores['BLEU 3'] += sentence_bleu([gt_process], res_process, weights=(0,0,1,0))/len(res)
            bleu_scores['BLEU 1'] += sentence_bleu([gt_process], res_process, weights=(0,0,0,1))/len(res)
            bleu_scores['BLEU S'] += sentence_bleu([gt_process], res_process)/len(res)
            
    else:
        for i in range(len(res)):
            bleu_scores['BLEU 1'] += sentence_bleu([gt[i].split()], res[i].split(), weights=(1,0,0,0))/len(res)
            bleu_scores['BLEU 2'] += sentence_bleu([gt[i].split()], res[i].split(), weights=(0,1,0,0))/len(res)
            bleu_scores['BLEU 3'] += sentence_bleu([gt[i].split()], res[i].split(), weights=(0,0,1,0))/len(res)
            bleu_scores['BLEU 1'] += sentence_bleu([gt[i].split()], res[i].split(), weights=(0,0,0,1))/len(res)
            bleu_scores['BLEU S'] += sentence_bleu([gt[i].split()], res[i].split())/len(res)
            
    return bleu_scores


def compute_scores(gts, res, mode, threshold):
    scores = {}
    bleu_scores = compute_bleu(gts, res, mode, threshold)
    scores = scores.update(bleu_scores)
    return scores
