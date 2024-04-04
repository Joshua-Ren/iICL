# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 15:12:09 2023

@author: YIREN
"""

import os
import time

class text_logger():
    def __init__(self, file_name, exp_path, silence=True):
        self.file_name = file_name
        self.exp_path = exp_path
        self.silence = silence
        
        self.file_path = os.path.join(self.exp_path, self.file_name+'.txt')
        if not os.path.exists(self.exp_path):
            os.makedirs(self.exp_path)
        self.init_logger()
        
    def write_to_file(self, text=None):
        if text is not None:
            if not self.silence:
                print(text)
            with open(self.file_path,"a") as f:
                f.write(text)
                f.write("\n")
        
    def init_logger(self):
        self.write_to_file("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        self.write_to_file("@@@@@@@@@@@@@@@@@@ " + time.asctime() +" @@@@@@@@@@@@@@@@@@")
        self.write_to_file("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        
    
    def msg_to_gpt(self, msg):
        self.write_to_file("\n## TO GPT:")
        self.write_to_file(msg)
        
    def msg_from_gpt(self,msg):
        self.write_to_file("\n## FEEDBACK:")
        self.write_to_file(msg)