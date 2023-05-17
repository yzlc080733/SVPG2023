# -*- coding: utf-8 -*-
import torch
import random

# Memory classes




def print_mem_info(input_string=''):
    print('\033[95mMEMORY_INFO|\033[0m', input_string)


# ~~~~~~~~~~~~~~~~~~~~RL Episode~~~~~~~~~~~~~~~~~~~~
class MemoryBuffer:
    def __init__(self, s_size, a_size, memory_size=1000, dev=torch.device('cpu'), batch_size=-1):
        # Settings
        self.memory_size = memory_size
        self.dev = dev
        self.for_rwta_flag = False
        self.batch_size = batch_size
        # Buffer
        self.s1 = torch.zeros([memory_size, s_size], device=self.dev)
        self.s2 = torch.zeros([memory_size, s_size], device=self.dev)
        
        self.model_output = torch.zeros([memory_size, a_size], device=self.dev)
        self.a = torch.zeros([memory_size, a_size], device=self.dev)

        self.a_logprob = torch.zeros([memory_size,], device=self.dev)
        self.r = torch.zeros([memory_size,], device=self.dev)
        self.done = torch.zeros([memory_size,], device=self.dev)
        # Pointer
        self.pointer, self.current_size = 0, 0

    def reset(self):
        self.pointer, self.current_size = 0, 0

    def init_for_rwta(self, q_size, v_size):
        self.for_rwta_flag = True
        self.q_has = torch.zeros([self.memory_size, q_size], device=self.dev)
        self.v_ha = torch.zeros([self.memory_size, v_size], device=self.dev)

    def add_transition(self, s1, model_output, a, a_log, r, s2, done, **kwargs):
        batch_size = s1.shape[0]
        if batch_size > self.memory_size:
            print_mem_info('Error batch too large')
        p1 = self.pointer
        p2 = min(self.pointer + batch_size, self.memory_size)
        p3 = p2 % self.memory_size
        p4 = p3 + batch_size - (p2 - p1)
        # part 1
        self.s1[p1:p2, :] = s1[0:(p2 - p1)]
        self.s2[p1:p2, :] = s2[0:(p2 - p1)]
        self.model_output[p1:p2, :] = model_output[0:(p2 - p1)]
        self.a[p1:p2, :] = a[0:(p2 - p1)]
        self.a_logprob[p1:p2] = a_log[0:(p2 - p1)]
        self.r[p1:p2] = r[0:(p2 - p1)]
        self.done[p1:p2] = done
        # part 2
        self.s1[p3:p4, :] = s1[(p2 - p1):]
        self.s2[p3:p4, :] = s2[(p2 - p1):]
        self.model_output[p3:p4, :] = model_output[(p2 - p1):]
        self.a[p3:p4, :] = a[(p2 - p1):]
        self.a_logprob[p3:p4] = a_log[(p2 - p1):]
        self.r[p3:p4] = r[(p2 - p1):]
        self.done[p3:p4] = done
        # Pointer
        self.current_size = min(self.current_size + batch_size, self.memory_size)
        self.pointer = (self.pointer + batch_size) % self.memory_size
        # RWTA
        if self.for_rwta_flag:
            q_has = kwargs.get('q_has')
            v_ha = kwargs.get('v_ha')
            self.q_has[p1:p2, :] = q_has[0:(p2 - p1)]
            self.v_ha[p1:p2, :] = v_ha[0:(p2 - p1)]
            self.q_has[p3:p4, :] = q_has[(p2 - p1):]
            self.v_ha[p3:p4, :] = v_ha[(p2 - p1):]

    def get_batch(self):
        if self.batch_size < 0:
            s1 = torch.clone(self.s1[:self.current_size, :])
            s2 = torch.clone(self.s2[:self.current_size, :])
            model_output = torch.clone(self.model_output[:self.current_size, :])
            a = torch.clone(self.a[:self.current_size, :])
            a_logprob = torch.clone(self.a_logprob[:self.current_size])
            r = torch.clone(self.r[:self.current_size])
            done = torch.clone(self.done[:self.current_size])
            if self.for_rwta_flag:
                q_has = torch.clone(self.q_has[:self.current_size, :])
                v_ha = torch.clone(self.v_ha[:self.current_size, :])
                return s1, s2, model_output, a, a_logprob, r, done, q_has, v_ha
            else:
                return s1, s2, model_output, a, a_logprob, r, done
        else:
            sample_list = random.sample(range(0, self.current_size), min(self.batch_size, self.current_size))
            s1 = torch.clone(self.s1[sample_list, :])
            s2 = torch.clone(self.s2[sample_list, :])
            model_output = torch.clone(self.model_output[sample_list, :])
            a = torch.clone(self.a[sample_list, :])
            a_logprob = torch.clone(self.a_logprob[sample_list])
            r = torch.clone(self.r[sample_list])
            done = torch.clone(self.done[sample_list])
            if self.for_rwta_flag:
                q_has = torch.clone(self.q_has[sample_list, :])
                v_ha = torch.clone(self.v_ha[sample_list, :])
                return s1, s2, model_output, a, a_logprob, r, done, q_has, v_ha
            else:
                return s1, s2, model_output, a, a_logprob, r, done




if __name__ == "__main__":
    memory = MemoryBuffer(s_size=784, a_size=10)
    print_mem_info('Memory start')
    



print('\033[91mFINISH: memory\033[0m')
