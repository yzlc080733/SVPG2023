# -*- coding: utf-8 -*-

import math
import numpy as np
import torch
import random
import skimage
import cv2

from vizdoom import DoomGame
from vizdoom import Mode
from vizdoom import ScreenFormat
from vizdoom import ScreenResolution
from vizdoom import AutomapMode

# ENV vizdoom HealthGathering


def print_info(input_string):
    INFO_MESSAGE = '\033[92mVIZDOOM_ENV_INFO|\033[0m'
    print(INFO_MESSAGE, input_string)


def check_data_type(input_variable):
    output_name = None
    if type(input_variable) == np.ndarray:
        output_name = 'cpu'
    elif type(input_variable) == torch.Tensor:
        if input_variable.device==torch.device(type='cpu'):
            output_name = 'torch_cpu'
        else:
            output_name = 'torch_gpu'
    return output_name


def get_onehot_from_numpy(array, num_class):
    if len(array.shape) != 1:
        print_mnist_info('Error in array shape (onehot conversion)')
        return None
    else:
        output_onehot = np.zeros([array.shape[0], num_class], dtype=np.float32)
        output_onehot[np.arange(array.shape[0]), array] = 1
        return output_onehot




class VIZDOOM_HEALTHGATHERING:
    def __init__(self, show_automap=False, show_window=False):
        # -----settings----------
        self.max_step_num = 525
        self.frame_repeat = 4
        self.SHOW_AUTOMAP = show_automap
        self.SHOW_WINDOW = show_window
        # -----init--------------
        self.init_params()
        self.init_game()
        self.dev = torch.device('cpu')

    def init_params(self):
        self.RESOLUTION = (60, 80)
        self.ACTIONS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]]
        self.state_dimension = self.RESOLUTION[0] * self.RESOLUTION[1]
        self.action_num = len(self.ACTIONS)
        print_info('ACTION NUMBER: %6d' % (self.action_num))

    def init_game(self):
        self.game = DoomGame()
        self.game.load_config('./env/doom_hg/health_gathering.cfg')
        self.game.set_doom_scenario_path('./env/doom_hg/health_gathering.wad')
        self.game.set_doom_map('map%02d' % 1)
        self.game.set_window_visible(self.SHOW_WINDOW)
        self.game.set_mode(Mode.PLAYER)
        if self.SHOW_AUTOMAP:
            print_info('Function not implemented')
            self.game.set_automap_buffer_enabled(True)
            self.game.set_automap_mode(AutomapMode.OBJECTS)
            self.game.add_game_args('+am_followplayer 1')
            self.game.add_game_args('+viz_am_scale 10')
            self.game.add_game_args('+viz_am_center 1')
            self.game.add_game_args('+am_backcolor 000000')
        self.game.init()

    def convert_all_to_torch_cpu(self):
        self.dev = torch.device('cpu')

    def convert_all_to_torch_gpu(self):
        self.dev = torch.device('cuda:0')

    def img_preprocess(self, img):
        img_view = np.moveaxis(img, 0, -1)
        img = cv2.resize(img_view, (self.RESOLUTION[1], self.RESOLUTION[0]), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32)
        img = img / 255
        return img

    def img_preprocess_RGB(self, img):
        img_view = np.moveaxis(img, 0, -1)
        # print('P1   ', img_view.shape)          # H*W*3
        img = cv2.resize(img_view, (self.RESOLUTION[1], self.RESOLUTION[0]), interpolation=cv2.INTER_LINEAR)
        # print('P2   ', img.shape)               # H*W*3
        img = img.astype(np.float32)
        img = img / 255
        img = np.moveaxis(img, 2, 0)
        # print('P2   ', img.shape)               # 3*H*W
        return img

    def img_preprocess_with_reshape(self, img):
        img_process = self.img_preprocess(img)
        return img_process.reshape([-1])

    def state_img_to_tensor(self, state_image):
        state_handle = np.copy(np.expand_dims(state_image, axis=0))
        state_cuda = torch.FloatTensor(state_handle).to(self.dev)
        return state_cuda

    def init_train(self):
        self.game.new_episode()
        self.mode = 'train'
        self.gather_num, self.step_num = 0, 0

    def init_val(self):
        self.game.new_episode()
        self.mode = 'val'
        self.gather_num, self.step_num = 0, 0

    def init_test(self):
        self.game.new_episode()
        self.mode = 'test'
        self.gather_num, self.step_num = 0, 0

    def get_observation(self):
        state_full = self.game.get_state()
        s = self.img_preprocess_with_reshape(state_full.screen_buffer)
        s_expand = np.copy(np.expand_dims(s, axis=0))
        s_tensor = torch.FloatTensor(s_expand).to(self.dev)
        return s_tensor

    def get_health_v(self):
        state_full = self.game.get_state()
        health_v = state_full.game_variables[0]
        return health_v

    def get_train_observation(self, **kwargs):
        s_tensor = self.get_observation()
        return s_tensor

    def get_val_observation(self, **kwargs):
        s_tensor = self.get_observation()
        return s_tensor

    def get_test_observation(self, noise_type='none', noise_param=0, **kwargs):
        s_tensor = self.get_observation()
        if noise_type == 'none':
            return s_tensor
        test_image_noise = s_tensor[0].cpu().numpy()
        temp_img = test_image_noise.reshape(self.RESOLUTION)
        # working ...
        if noise_type == 'gaussian':
            temp_img = skimage.util.random_noise(temp_img, mode='gaussian', seed=None, clip=True, var=noise_param**2)
        elif noise_type == 'pepper':
            temp_img = skimage.util.random_noise(temp_img, mode='pepper', seed=None, clip=True, amount=noise_param)
        elif noise_type == 'salt':
            temp_img = skimage.util.random_noise(temp_img, mode='salt', seed=None, clip=True, amount=noise_param)
        elif noise_type == 's&p':
            temp_img = skimage.util.random_noise(temp_img, mode='s&p', seed=None, clip=True, amount=noise_param, salt_vs_pepper=0.5)
        elif noise_type == 'gaussian&salt':
            temp_img = skimage.util.random_noise(temp_img, mode='gaussian', seed=None, clip=True, var=0.05**2)
            temp_img = skimage.util.random_noise(temp_img, mode='salt', seed=None, clip=True, amount=noise_param)
        test_image_noise = temp_img.reshape([-1])
        s_tensor_noise = self.state_img_to_tensor(test_image_noise)
        return s_tensor_noise

    def make_action(self, action):
        # -----old---------------
        old_health_value = self.get_health_v()
        # -----make action-------
        action_index = torch.argmax(action).item()
        env_reward = self.game.make_action(self.ACTIONS[action_index], self.frame_repeat)
        # -----new state---------
        done = self.game.is_episode_finished()
        player_dead = self.game.is_player_dead()
        # -----reward------------
        if player_dead:
            r = -50
            done_flag = 1
        elif done:
            r = 1
            done_flag = 1
        else:
            r = 1
            done_flag = 0
        if done_flag == 0:
            new_health_value  = self.get_health_v()
            if new_health_value > old_health_value:
                r = 10
                self.gather_num += 1
        self.step_num += 1
        # -----output------------
        self.done_signal = done_flag
        reward = torch.Tensor([r]).to(self.dev)
        return reward, [self.step_num, self.gather_num]
        # return reward, [self.gather_num]
        
        




if __name__ == "__main__":
    pass
    


print('\033[91mFINISH: env_mnist\033[0m')

