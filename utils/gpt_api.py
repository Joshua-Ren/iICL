# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 14:54:37 2023

@author: YIREN
"""

from openai import OpenAI
import anthropic
import openai
import toml
import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage



class multi_turn_chatgpt():
    def __init__(self, model='gpt-3.5-turbo', temperature=0.7, top_p=1, logger=None, game_description=None):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.logger = logger
        self.game_description = game_description
        self.history = []
        self.history_round = 0
        self.msg = " "
        
        if model.startswith('gpt'):
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        elif model.startswith('claude'):
            self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        elif model.startswith('mistral'):
            self.client = MistralClient(api_key=os.environ.get("MISTRAL_API_KEY"))

        if self.game_description is None:
            self.game_description = "You are a helpful assistant."
        
    def call_chatgpt(self, msg_to_api, fake_response=None, logprobs=False, top_logprobs=None, lookback_round=0, update_hist=True):

        # ----------- Prepare the historical message if needed
        if lookback_round == 0 or len(self.history)==0:
            self.msg = [{"role": "user", "content": msg_to_api}]
        else:
            hist_msg = self._select_history(lookback_round)
            self.msg = hist_msg
            self.msg.append({"role": "user", "content": msg_to_api})
        
            # ------- For GPT model, system info is attached at the beginning of self.msg
        if self.model.startswith('gpt'):
            self.msg = [{"role": "system", "content": self.game_description}] + self.msg

        # ----------- Get response if fake_response is not given
        if self.logger is not None:
            self.logger.msg_to_gpt(msg_to_api)  
            
        # ------- Update the history for every call
        if update_hist:
            self._update_history(msg_to_api, role="user")
        if fake_response is not None:
            receive_msg = fake_response
            cnt_tokens = (0,0,0)
        else:
            if self.model.startswith('gpt'):
                # ------------- API for gpt-xxx
                response = self.client.chat.completions.create(
                    model = self.model,
                    messages = self.msg,
                    temperature=self.temperature,
                    top_p = self.top_p,         
                    logprobs = logprobs,
                    top_logprobs = top_logprobs
                    )
                receive_msg = response.choices[0].message.content
                cnt_tokens = (response.usage.completion_tokens, 
                            response.usage.prompt_tokens, 
                            response.usage.total_tokens)
            elif self.model.startswith('claude'):
                response = self.client.messages.create(
                    model = self.model,
                    system = self.game_description,
                    messages = self.msg,
                    max_tokens = 2048,
                    temperature=self.temperature,
                    top_p = self.top_p,         
                    )     
                receive_msg = response.content[0].text
                cnt_tokens = (response.usage.output_tokens,
                              response.usage.input_tokens,
                              response.usage.output_tokens+response.usage.input_tokens)
            elif self.model.startswith('mistral'):
                response = self.client.chat(
                    model = self.model,
                    messages = self.msg,
                    temperature=self.temperature,
                    top_p = self.top_p, 
                    max_tokens = 2048,
                    )
                receive_msg = response.choices[0].message.content
                cnt_tokens = (response.usage.completion_tokens,
                              response.usage.prompt_tokens,
                              response.usage.total_tokens)
            else:
                print('Only support gpt, claude and mistral')
                raise NotImplementedError
            
        if update_hist:
            self._update_history(receive_msg, role="assistant")
            self.history_round += 1
        
        if logprobs and self.model.startswith('gpt'):
            receive_probs = response.choices[0].logprobs
        else:
            receive_probs = None
        
        if self.logger is not None:
            self.logger.msg_from_gpt(receive_msg)
        return receive_msg, receive_probs, cnt_tokens
    
    def _update_history(self, msg, role="user"):
        """
            Update the history of the chat.
            role can be "user" or "assistant"
        """
        if role=="user":
            tmp = {"role":"user","content":msg}
        elif role=="assistant":
            tmp = {"role":"assistant","content":msg}

        self.history.append(tmp)
    
    def _select_history(self, lookback_round):
        """
            Return the list of the required history, format is 
            [{"role": "user", "content":xxx}, {"role":"assistant","content":xxx}, 
             {"role": "user", "content":xxx}, {"role":"assistant","content":xxx}]
        """
        if lookback_round > self.history_round:
            lookback_round = self.history_round
        return self.history[-2*lookback_round:]
    