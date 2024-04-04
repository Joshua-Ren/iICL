# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 21:39:13 2024

@author: YIREN
"""
import json
import sys
sys.path.append('../')
from utils.h_and_d import h_x
import os
import numpy as np

def eval_feedback_h(h, fb_h, N):
    '''
        How many correct predicted rules
    '''
    corr_cnt = 0
    for key in h.keys():
        if key in fb_h.keys():
            if fb_h[key] == h[key]:
                corr_cnt += 1
    return corr_cnt, int(corr_cnt==N)
    
def convert_chatcompl_to_json(list_probs):
    out_json_list = []
    for k in range(len(list_probs)):
        fb_probs = list_probs[k]
        json_list = []
        chat_compl_logits = fb_probs.content
        for i in range(len(chat_compl_logits)):
            tmp_json = {}
            tmp_compl = chat_compl_logits[i]
            tmp_json['token'] = tmp_compl.token
            tmp_json['logprob'] = tmp_compl.logprob
            tmp_json['top_logprobs'] = []
            for j in range(len(tmp_compl.top_logprobs)):
                top_dicts = {}
                top_dicts['token'] = tmp_compl.top_logprobs[j].token
                top_dicts['logprob'] = tmp_compl.top_logprobs[j].logprob
                top_dicts['bytes'] = tmp_compl.top_logprobs[j].bytes
                tmp_json['top_logprobs'].append(top_dicts)
            json_list.append(tmp_json)
        out_json_list.append(json_list)
    return out_json_list
    
    
def dstr_to_pairs(d_str):
    tmp_str = d_str.split('\n')
    while "" in tmp_str:
        tmp_str.remove("")
    data_pairs = []
    for s in range(int(len(tmp_str)*0.5)):
        tmp_input = tmp_str[2*s].split(': ')[1].split(', ')
        tmp_output = tmp_str[2*s+1].split(': ')[1]
        data_pairs.append((tmp_input, tmp_output))
    return data_pairs

def count_corr_d0_pairs(d_pairs, rule):
    corr_cnt, all_cnt = 0, 0
    for x,y in d_pairs:
        all_cnt += 1
        if h_x(x, rule)==y:
            corr_cnt += 1
    return corr_cnt, all_cnt

def eval_get_corr_d0(d0_str, results_read):
    d0_corr_list = []
    d0_pairs = dstr_to_pairs(d0_str)
    GEN = len(results_read['rules_refine'])
    for i in range(GEN):
    #     if results_read['rules_refine'][i].startswith('Rule'):
    #         rule = results_read['rules_refine'][i].split('Rule: ')[1]
    #     else:
    #         rule = results_read['rules_refine'][i].split('\n\n')[1].split('Rule: ')[1]
        rule = results_read['rules_refine'][i]
        if type(rule) is not dict:
            rule = json.loads(results_read['rules_refine'][i])
        corr_cnt, _ = count_corr_d0_pairs(d0_pairs, rule)
        d0_corr_list.append(corr_cnt)
    return d0_corr_list

def get_d0_results(results_read, old_metric=True):
    # ----------- How many correct d0 each ht can predict
    d0_str = results_read['d_sampled'][0]   # Get d0 from the log
    d0_corr_list = eval_get_corr_d0(d0_str, results_read)  # How many d0 each ht can correctly predict
    # ----------- How many screen:off in each ht
    ht_screenoff_list = []
    # ----------- How many ht == h*
    h_tgt = results_read['rules'][0]
    h_tgt['screen'] = 'off'
    ht_tgt_list = []
    for i in range(len(results_read['rules_refine'])):
        tmp_screenoff = int(results_read['rules_refine'][i]['screen']=='off')
        ht_screenoff_list.append(tmp_screenoff)
        if old_metric:
            ht_tgt_list.append(int(results_read['rules_refine'][i]==h_tgt))  # old version
        else:
            if d0_corr_list[i]==8 and tmp_screenoff==1:
                ht_tgt_list.append(1)
            else:
                ht_tgt_list.append(0)       
    return d0_corr_list, ht_screenoff_list, ht_tgt_list

def regularize_results(results_read):    
    for i in range(len(results_read['rules'])):
        if type(results_read['rules'][i]) != dict:
            try:
                results_read['rules'][i] = json.loads(results_read['rules'][i].split("Rule: ")[1])
            except:
                results_read['rules'][i] = json.loads(results_read['rules'][i].split("Rule: ")[1].split('\n')[0])

    for i in range(len(results_read['rules_refine'])):
        if type(results_read['rules_refine'][i]) != dict:
            results_read['rules_refine'][i] = json.loads(results_read['rules_refine'][i])      
    return results_read
   
    

def cal_screen_off_hbar(rule_list, h_bar):
    cnt_screen, cnt_h = 0, 0
    for r in rule_list:
        if r['screen']=='off':
            cnt_screen += 1
        if r==h_bar:
            cnt_h += 1
    return cnt_screen, cnt_h, len(rule_list)

def updata_stats(stats, results_read, AVG_BACK=1, old_metric=True):
    d0_str = results_read['d_sampled'][0]
    d0_corr_list = eval_get_corr_d0(d0_str, results_read)
    h_bar = results_read['rules'][0]
    h_bar['screen']='off'
    cnt_screen_start, cnt_h, all_cnt = cal_screen_off_hbar(results_read['rules_refine'][:0], h_bar)
    cnt_screen, cnt_h, all_cnt = cal_screen_off_hbar(results_read['rules_refine'][-AVG_BACK:], h_bar)
    if not old_metric:
        _, _, ht_tgt_list = get_d0_results(results_read, old_metric=False)
        cnt_h = ht_tgt_list
    cnt_corr_d0 = np.sum(d0_corr_list[-AVG_BACK:])
    all_cnt_corr_d0 = AVG_BACK*8
    stats["cnt_screen_start"].append(cnt_screen_start)
    stats["cnt_screen_end"].append(cnt_screen)
    stats["cnt_h"].append(cnt_h)
    stats["cnt_all"].append(all_cnt)
    stats["cnt_d0"].append(cnt_corr_d0)
    stats["cnt_d0_all"].append(all_cnt_corr_d0)

def regularize_results(results_read):    
    for i in range(len(results_read['rules'])):
        if type(results_read['rules'][i]) != dict:
            results_read['rules'][i] = json.loads(results_read['rules'][i].split("Rule: ")[1])

    for i in range(len(results_read['rules_refine'])):
        if type(results_read['rules_refine'][i]) != dict:
            results_read['rules_refine'][i] = json.loads(results_read['rules_refine'][i])  
    return results_read

def get_results_read(exp_path_load):
    #save_path = os.path.join(exp_path_load, 'prob_list_all.json') 
    save_path2 = os.path.join(exp_path_load, 'other_results_all.json')
    #prob_list_read = json.load( open( save_path ))
    results_read = json.load(open(save_path2))
    return results_read
    

