# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 21:18:48 2024

@author: YIREN
"""
import sys
sys.path.append("..")
from utils.h_and_d import h_x


def gen_hd_prompt(data, ask_rule):
    # ---------- prompt_type can only be 'get_rule', or 'get_example'
    tmp = 'You now have more examples, generate a rule that maps the all the inputs (including those in previous rounds) to their corresponding outputs.\n\n%sPlease format your rule in the following format.\n%s\nWe only have five objects, they are ["A", "B", "C", "D", "screen"].\nPlease ONLY return the rule!\nPlease ONLY return the rule!\nPlease ONLY return the rule!\n'%(data, ask_rule) 
    return tmp

def gen_dh_prompt(M, rule):
    example = 'Input: obj1, obj2, ...\nOutput: on/off/und\n\nInput: obj1, obj2, ...\nOutput: on/off/und\n'
    re_state = 'Remember the input is a list of objects. If any objects with status on in the input, the output should be on. If all objects in the input are off, the output is off. If only objects with off and und in the list, the output should be undetermined (und for short).'
    #tmp = 'The rule you provided is %s. Based on this rule, can you give %d examples with different inputs those are unseen before? %s Please ONLY return the input-output pairs strictly following this format:\n\n%s'\
#%(rule, M, re_state, example)    # All gpt experiments are based on this one ... claude cannot use this
    tmp = 'The rule you provided is %s. Based on this rule, can you give %d more examples? %s Please ONLY return the input-output pairs strictly following this format:\n\n%s'\
%(rule, M, re_state, example)
    return tmp

def gen_data_prompt(d, need_stat=True):
    '''
        Convert data d to string (attached to the prompt)
    '''
    tmp = ""
    for index, row in d.iterrows():
        if need_stat:
            tmp += 'Input: %s\nOutput: %s\n'%(row['obj_str'], row['stat'])
        else:
            tmp += 'Input: %s\nOutput: \n'%(row['obj_str'])
    return tmp


def gen_h_refine_prompt(h, data_str, rule_format):
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

    refine_flag = False
    feedback = ""
    refine_str = ""
    d_pairs = dstr_to_pairs(data_str)
    for (x,y) in d_pairs:
        if h_x(x, h) != y:
            x_str = ", ".join(x)
            refine_str += "Input: %s\nPredict output: %s\nGround truth output: %s\n\n"%(x_str, h_x(x, h), y)
    if len(refine_str)>0:
        refine_flag = True
        feedback = "The rule you give is %s. Applying it to the following inputs does not produce the expected outputs.\n\n%sPlease refine your rule based on the above feedback to make sure it applies to all the examples.\nPlease format your rule in the following format.\n%s\nPlease only return the rule without any explanations."%(h, refine_str, rule_format)
    else:
        refine_flag = False
        feedback = "The rule you provided is %s, which fits all examples, well done!"%h
    return feedback, refine_flag


def gen_h_search_prompt(h, data_str, rule_format):
    tmp = 'The rule you give is %s. Please verify whether it can explain the following examples. \
If not, refine your rule to make sure it applies to all the examples. \
If yes, return the correct rule.\n\n%s\n\
We only have five objects, they are ["A", "B", "C", "D", "screen"].\
\nPlease ONLY return a JSON following this format:\n\
-reason: str, how you refine your rule\n\
-Rule: str, in the format %s'%(h, data_str, rule_format[6:])    
    return tmp

