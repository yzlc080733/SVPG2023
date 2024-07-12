# -*- coding: utf-8 -*-

import numpy as np
import torch
import sys
import os
sys.path.insert(0, os.getcwd() + '/env')
import gym


# ENV InvertedPendulum

# default_length = 1.5
# default_thickness = 0.05


def print_info(input_string):
    print('\033[92mGYMIP_ENV_INFO|\033[0m', input_string)


class GymIP:
    def __init__(self, train_xml_name='inverted_pendulum_ChangeThk_0.050000.xml',
                 dev=torch.device('cpu')):
        # Settings
        self.max_step_num = 200
        self.state_dimension = 4
        self.action_num = 5
        self.dev = dev
        self.env = None
        # Variables
        self.mode, self.step_num, self.done_signal = None, None, None
        self.state_original, self.state_processed = None, None
        self.train_xml_name = train_xml_name
        print_info('ACTION NUMBER: %6d' % self.action_num)
        print_info('XML: %s' % self.train_xml_name)

    def state_to_tensor(self, state):
        state_handle = np.copy(state)
        state_handle = state_handle * np.array([2.5, 5, 0.6, 0.8])      # [6, 5, 0.6, 0.8]
        state_handle = np.clip(state_handle, -1, 1)
        # state_handle = np.clip(state_handle, -1, 1) * 0.5 + 0.5
        # state_cuda = torch.FloatTensor(np.expand_dims(state_handle, axis=0)).to(self.dev)
        # return state_cuda
        return state_handle     # NUMPY ARRAY FOR BASELINE 3

    def action_index_handler(self, action_index_in):
        return action_index_in * 6 / (self.action_num - 1) - 3

    def init_train(self):
        if self.env is not None:
            self.env.close()
        self.env = gym.make('InvertedPendulum-v4', render_mode=None, self_xml=self.train_xml_name)
        self.state_original = self.env.reset()[0]
        self.step_num = 0

    def init_val(self):
        if self.env is not None:
            self.env.close()
        self.env = gym.make('InvertedPendulum-v4', render_mode=None, self_xml=self.train_xml_name)
        self.state_original = self.env.reset()[0]
        self.step_num = 0

    def init_test(self, variation_type='none', variation_param=0):
        if self.env is not None:
            self.env.close()
        if variation_type == 'none':
            self.env = gym.make('InvertedPendulum-v4', render_mode=None, self_xml=self.train_xml_name)
        elif variation_type in ['length', 'thick', 'union']:
            if variation_type == 'length':
                file_name_try = 'inverted_pendulum_ChangeLen_%f' % variation_param + '.xml'
            elif variation_type == 'thick':
                file_name_try = 'inverted_pendulum_ChangeThk_%f' % variation_param + '.xml'
            else:                   # both
                file_name_try = 'inverted_pendulum_ChangeBoth_%f' % variation_param + '.xml'
            self.env = gym.make('InvertedPendulum-v4', render_mode=None, self_xml=file_name_try)
        else:
            print_info('Error in variation type')
        self.state_original = self.env.reset()[0]
        self.step_num = 0

    def get_observation(self):
        self.state_processed = self.state_to_tensor(self.state_original)
        return self.state_processed

    def get_train_observation(self, **kwargs):
        return self.get_observation()

    def get_val_observation(self, **kwargs):
        return self.get_observation()

    def get_test_observation(self, noise_type='none', noise_param=0, **kwargs):
        s_tensor = self.get_observation()       # a transformed observation, in [0, 1]
        if noise_type == 'none':
            return s_tensor
        s_tensor = torch.from_numpy(s_tensor)       # FOR BASELINE 3
        s_tensor = s_tensor * 0.5 + 0.5
        if noise_type == 'gaussian':
            s_tensor.add_(torch.randn(s_tensor.size()).to(self.dev) * noise_param)
        else:  # uniform
            s_tensor.add_((torch.rand(s_tensor.size()).to(self.dev) - 0.5) * 2 * noise_param)
        s_tensor = s_tensor * 2 - 1
        s_tensor = s_tensor.numpy()                 # FOR BASELINE 3
        return s_tensor

    def make_action(self, action):
        # Action
        action_index = action           # FOR BASELINE 3
        action_value = self.action_index_handler(action_index)
        s2, r, done, info, _ = self.env.step([action_value])
        # Reward
        r = 1
        reward = torch.FloatTensor(np.array([r])).to(self.dev)
        # Done signal
        if done or self.step_num == 199:
            done_flag = 1
        else:
            done_flag = 0
        self.done_signal = done_flag
        # Transfer
        self.state_original = np.copy(s2)
        self.step_num += 1
        return reward, self.state_to_tensor(s2), [self.step_num]


if __name__ == "__main__":
    env = GymIP(dev=torch.device('cuda:0'))

print('\033[91mFINISH: env_gymip\033[0m')
