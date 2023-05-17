# -*- coding: utf-8 -*-
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F




# Memory classes




def print_mem_info(input_string=''):
    INFO_MESSAGE = '\033[95mMEMORY_INFO|\033[0m'
    print(INFO_MESSAGE, input_string)



# ~~~~~~~~~~~~~~~~~~~~RL Episode~~~~~~~~~~~~~~~~~~~~
class MemoryList:
    def __init__(self, gamma=0.97):
        self.gamma = gamma
        self.reset()

    def reset(self):
        self.s1_list = []
        self.a_output_list = []
        self.a_logprob_list = []
        self.a_onehot_list = []
        self.s2_list = []
        self.r_list = []
        self.q_list = []
        self.other_list = []

    def add_transition(self, s1, a_output, a_logprob, a_onehot, r, s2, other):
        self.s1_list.append(s1)
        self.a_output_list.append(a_output)
        self.a_logprob_list.append(a_logprob)
        self.a_onehot_list.append(a_onehot)
        self.s2_list.append(s2)
        self.r_list.append(r)
        self.other_list.append(other)

    def tune_reward(self, reward_normalization=True):
        queue_length = len(self.r_list)
        if reward_normalization:
            r = torch.cat(self.r_list)
            r = (r - r.mean()) / (r.std() + 1e-5)
            for i in range(queue_length):
                self.r_list[i] = r[i]
        self.q_list = self.r_list.copy()
        for i in range(queue_length-2, -1, -1):
            self.q_list[i] = self.q_list[i] + self.gamma * self.q_list[i+1]

    def get_batch(self):
        s1 = torch.cat(self.s1_list)
        a_output = torch.cat(self.a_output_list)
        a_logprob = torch.cat(self.a_logprob_list)
        a_onehot = torch.cat(self.a_onehot_list)
        r = torch.stack(self.r_list)
        q = torch.stack(self.q_list)
        s2 = self.s2_list               # List of None
        other = self.other_list
        return s1, a_output, a_logprob, a_onehot, q, s2, other




# ~~~~~~~~~~~~~~~~~~~~Batch for classification~~~~~~
class MemoryBatch:
    def __init__(self):
        self.reset()

    def reset(self):
        self.s1 = None
        self.a_output = None
        self.a_logprob = None
        self.a_onehot = None
        self.r = None
        self.s2 = None
        self.other = None

    def add_transition(self, s1, a_output, a_logprob, a_onehot, r, s2, other):
        self.s1 = s1
        self.a_output = a_output
        self.a_logprob = a_logprob
        self.a_onehot = a_onehot
        self.r = r
        self.s2 = s2
        self.other = other

    def tune_reward(self, **kwargs):
        self.q = self.r

    def get_batch(self):
        # print('>>>   ', self.s1.shape, self.a_output.shape, self.r.shape)
        return self.s1, self.a_output, self.a_logprob, self.a_onehot, self.r, self.s2, self.other







if __name__ == "__main__":
    print_mem_info('MLP model start')
    



print('\033[91mFINISH: memory\033[0m')


