# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 11:32:17 2024

@author: YIREN
"""

import numpy as np
import torch
import random
import pandas as pd
import copy


STATES = ['on','off', 'und']

# ======================= Operations about h

def gen_hstar_rnd(OBJECTS, PROBS):
    # --------- Randomly generate h*
    while True:
        h_star = {}
        on_cnt, off_cnt, ud_cnt = 0, 0, 0
        for o in OBJECTS:
            stat = np.random.choice(STATES, size=1, p=PROBS)[0]
            h_star[o] = stat
            if stat==STATES[0]:
                on_cnt += 1
            elif stat==STATES[1]:
                off_cnt += 1
            elif stat==STATES[2]:
                ud_cnt += 1
        if on_cnt!=0 and off_cnt!=0:
            return h_star

def gen_hstar_given_status(status, OBJECTS):
    # --------- Generate h* given status
    h_tmp = {}
    for j in range(len(OBJECTS)):
        h_tmp[OBJECTS[j]] = status[j]
    return h_tmp

def cnt_of_status(stat_list):
    # -------- Count how many on/off/und in the given stat_list
    n_on, n_off, n_und=0,0,0
    for s in stat_list:
        if s=='on':
            n_on += 1
        elif s=='off':
            n_off += 1
        else:
            n_und += 1
    return n_on, n_off, n_und

def h_x(input_list, h):
    # ---- y = h(x)
    on_cnt, off_cnt, ud_cnt = 0, 0, 0
    for ob in input_list:
        stat = h[ob]
        if stat==STATES[0]:
            on_cnt += 1
        elif stat==STATES[1]:
            off_cnt += 1
        else:
            ud_cnt += 1

    if on_cnt>0:
        return STATES[0]
    elif ud_cnt>0:
        return STATES[2]
    else:
        return STATES[1]

# ======================= Data generator
class data_generator():
    def __init__(self, h, N_test=10):
        self.h = h              # dictionary of mapping
        self.objects = list(self.h.keys())  # Names of objects, like ['A', 'B','C','D']
        self.N_test = N_test
        self.N = len(self.objects)
        
        self.data_df = self._gen_dataframe()


    def sample_d0(self, M):
        # ---- Sample M data pairs from training split
        return self.data_df[self.data_df['split']=='train'].sample(n=M)
    
    def sample_d0_under_hbar(self, M, h_bar):
        # ----- Sample M data pairs that can also explained by hbar
        tmp_df = copy.deepcopy(self.data_df)
        for idx, row in tmp_df.iterrows():
            if h_x(row['obj'], self.h) == h_x(row['obj'], h_bar):
                tmp_df.loc[idx, ['split']]=['overlap']
            else:
                tmp_df.loc[idx, ['split']]=['no']        
        return tmp_df[tmp_df['split']=='overlap'].sample(n=M)
    
    def get_all_test_samples(self):
        return self.data_df[self.data_df['split']=='test']
        
    def _gen_dataframe(self):
        N_SMP = 2**self.N-1
        N_Train = N_SMP - self.N_test
        tmp_mask = np.random.permutation(np.arange(0,N_SMP))
        train_idx = tmp_mask[:N_Train]
        data_df = pd.DataFrame(columns=['index','obj','obj_str','stat','split','size'])
        for bi in range(1, N_SMP+1):
            obj = self._get_obj_str(bi)
            obj_str = self._obj_to_str(obj)
            size = len(obj)
            stat = h_x(obj, self.h)
            if bi in train_idx:
                split = "train"
            else:
                split = "test"
            data_df.loc[bi] = [bi, obj, obj_str, stat, split, size]
        return data_df
    
    def _get_obj_str(self, index=1):
        # ------ Create list of inputs
        bi_idx = np.binary_repr(index)
        if len(bi_idx)<self.N:
            fil_ = ""
            for i in range(self.N-len(bi_idx)):
                fil_ += "0"
            sel_idx = fil_ + bi_idx
        else:
            sel_idx = bi_idx
            
        obj_list = []
        for i in range(len(sel_idx)):
            if sel_idx[i]=='1':
                obj_list.append(self.objects[i])
        return obj_list
    
    def _obj_to_str(self, obj):
        tmp = ""
        c_flag = False
        for s in obj:
            c_flag = True
            tmp += s
            tmp += ", "
        if c_flag:
            return tmp[:-2]
        else:
            return " "
        
if __name__ == "__main__":
    h_star = {'A': 'on', 'B': 'off', 'C': 'off', 'D': 'on', 'screen': 'und'}
    data_generator = data_generator(h_star, N_test=10)
    d0 = data_generator.sample_d0(M=5)
    test_data = data_generator.get_all_test_samples()
    
    
    
    
    
    
    
