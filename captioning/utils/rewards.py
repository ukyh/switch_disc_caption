from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
from collections import OrderedDict
import torch

import sys
try:
    sys.path.append("cider")
    from pyciderevalcap.ciderD.ciderD import CiderD
    from pyciderevalcap.cider.cider import Cider
    sys.path.append("coco-caption")
    from pycocoevalcap.bleu.bleu import Bleu
    from cider_btw import CiderBtw
except:
    print('cider or coco-caption missing')

CiderD_scorer = None
Cider_scorer = None
Bleu_scorer = None
CiderBtw_scorer = None
#CiderD_scorer = CiderD(df='corpus')

def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Cider_scorer
    Cider_scorer = Cider_scorer or Cider(df=cached_tokens)
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)
    global CiderBtw_scorer
    CiderBtw_scorer = CiderBtw_scorer or CiderBtw(df=cached_tokens)

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

def get_self_critical_reward(greedy_res, data_gts, gen_result, opt):
    batch_size = len(data_gts) 
    gen_result_size = gen_result.shape[0]
    seq_per_img = gen_result_size // len(data_gts) # gen_result_size  = batch_size * seq_per_img
    assert greedy_res.shape[0] == batch_size

    res = OrderedDict()
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(gen_result_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[gen_result_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(len(res))]
    res__ = {i: res[i] for i in range(len(res_))}
    gts_ = {i: gts[i // seq_per_img] for i in range(gen_result_size)}
    gts_.update({i+gen_result_size: gts[i] for i in range(batch_size)})
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts_, res_)
        print('Cider scores:', _)
    else:
        cider_scores = 0
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts_, res__)
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
    else:
        bleu_scores = 0
    scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores

    scores = scores[:gen_result_size].reshape(batch_size, seq_per_img) - scores[-batch_size:][:, np.newaxis]
    scores = scores.reshape(gen_result_size)

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards

def get_btw_score(gen_result, data_sim, opt, mode='raw'):
    batch_size = len(data_sim)      # data_sim: [np(sim_per_img * gts_per_img, 16) * batch]
    gen_result_size = gen_result.shape[0]   # gen_result: (batch_size * seq_per_img, 16) or np in the same shape
    seq_per_img = gen_result_size // len(data_sim)

    res = OrderedDict()
    if type(gen_result) is torch.Tensor:
        gen_result = gen_result.data.cpu().numpy()
    for i in range(gen_result_size):
        res[i] = [array_to_str(gen_result[i])]

    gts = OrderedDict()
    for i in range(len(data_sim)):
        gts[i] = [array_to_str(data_sim[i][j]) for j in range(len(data_sim[i]))]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(len(res))]
    gts_ = {i: gts[i // seq_per_img] for i in range(gen_result_size)}
    _, cider_scores = CiderD_scorer.compute_score(gts_, res_)   # np(batch_size * seq_per_img)
    print('CiderBtw raw score:', cider_scores.mean())
    if mode != 'raw':
        max_per_img = np.array(
            [cider_scores[i*seq_per_img: (i+1)*seq_per_img].max() for i in range(batch_size)]
        )   # np(batch)
        norm_term = np.repeat(max_per_img, seq_per_img) # np(batch * seq_per_img)
        btw_scores = opt.btw_lw - opt.btw_aw * cider_scores / (norm_term + 1e-8) # np(batch * seq_per_img)
        print('CiderBtw weight:', btw_scores.mean())
    else:
        btw_scores = cider_scores
    scores = np.repeat(btw_scores[:, np.newaxis], gen_result.shape[1], 1)  # np(batch * seq_per_img, 16)

    return scores

def get_btw_scst_reward(gen_result, greedy_res, data_gts, sim_data_gts, gts_btw, opt):
    batch_size = len(data_gts) 
    gen_result_size = gen_result.shape[0]
    seq_per_img = gen_result_size // len(data_gts) # gen_result_size = batch_size * seq_per_img
    gts_per_img = len(data_gts[0])
    assert greedy_res.shape[0] == batch_size
    assert len(data_gts) == len(sim_data_gts)
    assert len(data_gts) * gts_per_img == len(gts_btw)

    res = OrderedDict()
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(gen_result_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[gen_result_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]
    
    res_ = [{'image_id': i, 'caption': res[i]} for i in range(len(res))]
    gts_ = {i: gts[i // seq_per_img] for i in range(gen_result_size)}
    gts_.update({i+gen_result_size: gts[i] for i in range(batch_size)})
    btw_ = {i: gts_btw[(i // seq_per_img) * gts_per_img: (i // seq_per_img + 1) * gts_per_img][:, 0] for i in range(gen_result_size)}   # {0:np(gts_per_img), ...}
    btw_.update({i+gen_result_size: btw_[i * seq_per_img] for i in range(batch_size)})

    _, btw_cider_scores = CiderBtw_scorer.compute_score(gts_, res_, btw_) # np(batch_size * seq_per_img + batch_size)
    print('CiderBtw weighted score:', _)

    rewards = btw_cider_scores[:gen_result_size].reshape(batch_size, seq_per_img) - btw_cider_scores[-batch_size:][:, np.newaxis]
    rewards = rewards.reshape(gen_result_size)
    rewards = np.repeat(rewards[:, np.newaxis], gen_result.shape[1], 1)

    return rewards

def get_scores(data_gts, gen_result, opt):
    batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data_gts)

    res = OrderedDict()
    
    gen_result = gen_result.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]

    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(batch_size)]
    res__ = {i: res[i] for i in range(batch_size)}
    gts = {i: gts[i // seq_per_img] for i in range(batch_size)}
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_)
        print('Cider scores:', _)
    else:
        cider_scores = 0
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
    else:
        bleu_scores = 0

    scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores

    return scores

def get_self_cider_scores(data_gts, gen_result, opt):
    batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data_gts)

    res = []
    
    gen_result = gen_result.data.cpu().numpy()
    for i in range(batch_size):
        res.append(array_to_str(gen_result[i]))

    scores = []
    for i in range(len(data_gts)):
        tmp = Cider_scorer.my_self_cider([res[i*seq_per_img:(i+1)*seq_per_img]])
        def get_div(eigvals):
            eigvals = np.clip(eigvals, 0, None)
            return -np.log(np.sqrt(eigvals[-1]) / (np.sqrt(eigvals).sum())) / np.log(len(eigvals))
        scores.append(get_div(np.linalg.eigvalsh(tmp[0]/10)))

    scores = np.array(scores)

    return scores