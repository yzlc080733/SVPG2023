# -*- coding: utf-8 -*-

import math
import numpy as np
import random
import skimage
import torch
import os
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from ai2thor.util import metrics
import cv2
import random
import utils
import pickle



# ENV AI2THOR NAVIGATION FOR EXP SUPPLEMENTATION


if os.path.exists('/data/.ai2thor'):
    os.environ['HOME'] = '/data/'
else:
    os.environ['HOME'] = './'



def print_info(input_string):
    INFO_MESSAGE = '\033[92mAI2THOR_ENV_INFO|\033[0m'
    print(INFO_MESSAGE, input_string)



class AI2THOR_NAVIGATION:
    def __init__(self, show_window=False):
        # -----settings----------
        self.max_step_num = 200
        self.SHOW_WINDOW = show_window
        # self.SHOW_WINDOW = True
        # -----init--------------
        self.init_params()
        self.init_game()
        self.dev = torch.device('cpu')

    def convert_all_to_torch_cpu(self):
        self.dev = torch.device('cpu')

    def convert_all_to_torch_gpu(self):
        self.dev = torch.device('cuda:0')
    
    def init_params(self):
        self.RESOLUTION = (60, 80)
        self.ACTIONS = ['MoveAhead', 'RotateLeft', 'RotateRight', 'MoveLeft', 'MoveRight']
        self.state_dimension = self.RESOLUTION[0] * self.RESOLUTION[1]
        self.action_num = len(self.ACTIONS)
        print_info('ACTION NUMBER: %6d' % (self.action_num))

    def init_game(self):
        self.controller = Controller(
            agentMode='locobot',
            visibilityDistance=1.5,
            scene='FloorPlan_Train1_3',
            gridSize=0.15,
            movementGaussianSigma=0.005,
            snapToGrid=True,
            rotateStepDegrees=90,
            rotateGaussianSigma=0.5,
            renderDepthImage=False,
            renderInstanceSegmentation=False,
            width=self.RESOLUTION[1],
            height=self.RESOLUTION[0],
            fieldOfView=60,
            ######
            platform=CloudRendering,        # OFF-SCREEN RENDERING
        )
        self.controller.reset(scene='FloorPlan_Train7_5')
        ''' GENERATE STARTING POSITION LIST AND SAVE -- GRID SIZE 0.25
        self.positions_list = self.controller.step(action='GetReachablePositions').metadata['actionReturn']
        with open('./env_data_list.pkl', 'wb') as file_out:
            pickle.dump(self.positions_list, file_out)
        '''
        with open('./env_data_list.pkl', 'rb') as file_in:
            self.positions_list = pickle.load(file_in)
        self.train_index_list = list(range(0, len(self.positions_list), 10))
        self.valid_index_list = list(range(1, len(self.positions_list), 40))
        # self.train_index_list = [0]
        # self.valid_index_list = [0]
        print('----', len(self.train_index_list), len(self.valid_index_list))
        '''
            NOTE: LENGTHS == (30, 8)
        '''
        self.validation_index_pointer = 0
        self.test_index_pointer = 0

    def init_train(self):
        self.controller.reset(scene='FloorPlan_Train7_5')
        self.controller.step(action='Done')
        position_index = random.choice(self.train_index_list)
        position_to_teleport = self.positions_list[position_index]
        self.current_event = self.controller.step(action='Teleport', position=position_to_teleport, rotation=dict(x=0, y=270, z=0))
        self.mode = 'train'
        self.gather_num, self.step_num = 0, 0

    def init_val(self):
        self.controller.reset(scene='FloorPlan_Train7_5')
        self.controller.step(action='Done')
        position_index = self.valid_index_list[self.validation_index_pointer]
        self.validation_index_pointer = (self.validation_index_pointer + 1) % len(self.valid_index_list)
        position_to_teleport = self.positions_list[position_index]
        self.current_event = self.controller.step(action='Teleport', position=position_to_teleport, rotation=dict(x=0, y=270, z=0))
        self.mode = 'val'
        self.gather_num, self.step_num = 0, 0

    def init_test(self):
        self.controller.reset(scene='FloorPlan_Train7_5')
        self.controller.step(action='Done')
        position_index = self.valid_index_list[self.test_index_pointer]
        self.test_index_pointer = (self.test_index_pointer + 1) % len(self.valid_index_list)
        position_to_teleport = self.positions_list[position_index]
        self.current_event = self.controller.step(action='Teleport', position=position_to_teleport, rotation=dict(x=0, y=270, z=0))
        self.mode = 'test'
        self.gather_num, self.step_num = 0, 0

    def get_observation(self):
        cv2img = self.current_event.cv2img
        s = self.img_preprocess_with_reshape(cv2img)
        s_tensor = self.state_img_to_tensor(s)
        return s_tensor

    def img_preprocess_with_reshape(self, img):
        img_process = self.img_preprocess(img)
        return img_process.reshape([-1])

    def img_preprocess(self, img):
        '''
            img.shape == (60, 80, 3), maximum value=255
        '''
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32)
        img = img / 255
        return img

    def state_img_to_tensor(self, state_image):
        state_handle = np.copy(np.expand_dims(state_image, axis=0))
        state_cuda = torch.FloatTensor(state_handle).to(self.dev)
        return state_cuda
    
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

    def get_env_variables(self):
        # TARGET DISTANCE
        object_metadata = self.current_event.metadata['objects']
        found_num = 0
        found_index = None
        for find_i in range(len(object_metadata)):
            if object_metadata[find_i]['objectType'] == 'Television':
                found_num += 1
                found_index = find_i
        assert found_num == 1
        target_metadata = object_metadata[found_index]
        target_distance = target_metadata['distance']
        target_in_view = target_metadata['visible']

        # TARGET PATH DISTANCE
        # shortest_path = metrics.get_shortest_path_to_object_type(
        #     controller=self.controller, object_type='Television',
        #     initial_position=self.current_event.metadata['agent']['position'],
        # )
        # path_distance = metrics.path_distance(shortest_path)
        # ERROR MESSAGE
        action_flag = self.current_event.metadata['lastActionSuccess']
        return {'target_distance': target_distance,
                'target_in_view': target_in_view,
                'action_successful': action_flag,
            }
    
    def make_action(self, action):
        # -----old---------------
        old_env_variables = self.get_env_variables()
        old_distance = old_env_variables['target_distance']
        old_target_in_view = old_env_variables['target_in_view']
        # -----make action-------
        action_index = torch.argmax(action).item()
        # ['MoveAhead', 'RotateLeft', 'RotateRight', 'MoveLeft', 'MoveRight']
        action_string = self.ACTIONS[action_index]
        if action_string == 'MoveAhead':
            self.current_event = self.controller.step(action='MoveAhead')
        elif action_string == 'RotateLeft':
            self.current_event = self.controller.step(action='RotateLeft')
            self.current_event = self.controller.step(action='MoveAhead')
        elif action_string == 'RotateRight':
            self.current_event = self.controller.step(action='RotateRight')
            self.current_event = self.controller.step(action='MoveAhead')
        elif action_string == 'MoveLeft':
            self.current_event = self.controller.step(action='RotateLeft')
            self.current_event = self.controller.step(action='MoveAhead')
            self.current_event = self.controller.step(action='RotateRight')
        elif action_string == 'MoveRight':
            self.current_event = self.controller.step(action='RotateRight')
            self.current_event = self.controller.step(action='MoveAhead')
            self.current_event = self.controller.step(action='RotateLeft')
        else:
            raise Exception('Error in action selection')
        # -----new state---------
        new_env_variables = self.get_env_variables()
        new_distance = new_env_variables['target_distance']
        new_target_in_view = new_env_variables['target_in_view']
        new_action_flag = new_env_variables['action_successful']
        # -----reward------------
        if new_target_in_view == True:
            done_flag = True
            r = 50
        else:
            done_flag = False
            if new_action_flag == False:
                r = -5
            else:
                if new_distance < old_distance:
                    r = +1
                    self.gather_num += 1
                elif new_distance > old_distance:
                    r = -1
                else: r = 0
        self.step_num += 1
        # -----output------------
        self.done_signal = done_flag
        reward = torch.Tensor([r]).to(self.dev)
        ''' NOTE: self.gather_num is the number of approaching steps'''
        ''' NOTE: variable[0] step num is checked to save for larger values'''
        # CHECK
        if self.SHOW_WINDOW == True:
            utils.check_image(self.img_preprocess(self.current_event.cv2img))
            # utils.check_image(self.current_event.cv2img)
        return reward, [-self.step_num, self.gather_num]
        

        


if __name__ == "__main__":
    pass
    


print('\033[91mFINISH: env_ai2thor\033[0m')

