# -*- coding: utf-8 -*-

import math
import numpy as np
import torch
import random
import skimage
import os
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
import cv2
import random
import utils



# ENV ROBOTARM VISION-BASED CONTROL FOR EXP SUPPLEMENTATION



def print_info(input_string):
    INFO_MESSAGE = '\033[92mROBOTARM_ENV_INFO|\033[0m'
    print(INFO_MESSAGE, input_string)



class ROBOTARM_CONTROL:
    def __init__(self, show_window=False):
        # SETTINGS
        self.max_step_num = 60
        self.SHOW_WINDOW = show_window
        # self.SHOW_WINDOW = True
        self.SHOW_STATE_VARIABLES = False
        self.SAVE_STATE_IMAGE = False
        # INIT
        self.init_params()
        self.init_game()
        self.dev = torch.device('cpu')

    def convert_all_to_torch_cpu(self):
        self.dev = torch.device('cpu')

    def convert_all_to_torch_gpu(self, cuda_num):
        self.dev = torch.device('cuda:%d' % cuda_num)
    
    def init_params(self):
        self.RESOLUTION = (64, 64)
        self.ACTIONS = [0, 1, 2, 3, 9]
        self.state_dimension = self.RESOLUTION[0] * self.RESOLUTION[1]
        self.action_num = len(self.ACTIONS)
        print_info('ACTION NUMBER: %6d' % (self.action_num))

    def init_game(self):
        SCENE_FILE = os.path.join('./env_data/scene_jaco.ttt')
        self.pr = PyRep()
        if self.SHOW_WINDOW == True:
            self.pr.launch(SCENE_FILE, headless=False, responsive_ui=False)
        else:
            # self.pr.launch(SCENE_FILE, headless=False, responsive_ui=False)
            self.pr.launch(SCENE_FILE, headless=True)
        self.pr.start()
        # RANDOM TARGET POSITION
        # self.cuboid_position_list = [
        #     # RANDOMLY GENERATED
        #     [0.67, 0.07], [0.64, 0.07], [0.71, 0.23], [0.66, 0.13], [0.67, 0.09],
        #     [0.64, -0.01], [0.78, 0.02], [0.72, 0.20], [0.74, 0.22], [0.72, 0.08],
        #     [0.67, 0.05], [0.71, 0.12], [0.75, 0.16], [0.74, 0.11], [0.77, 0.02],
        #     [0.72, -0.03], [0.77, 0.09], [0.62, 0.02], [0.66, 0.02], [0.76, 0.07],
        # ]
        self.cuboid_position_list = [
            # RANDOMLY GENERATED
            [0.78, 0.08],[0.71, 0.03],[0.69, -0.03],[0.65, -0.01],[0.76, 0.22],
            [0.78, 0.05],[0.74, 0.04],[0.62, 0.18],[0.64, -0.04],[0.71, 0.07],
            [0.72, -0.01],[0.65, 0.16],[0.73, 0.04],[0.69, 0.05],[0.65, 0.15],
            [0.70, 0.19],[0.78, 0.21],[0.68, 0.24],[0.69, 0.04],[0.66, 0.01],
            [0.77, 0.11],[0.76, 0.03],[0.80, 0.11],[0.62, 0.18],[0.71, 0.18],
            [0.66, -0.01],[0.77, 0.19],[0.76, -0.02],[0.79, 0.04],[0.77, 0.09],
            [0.74, 0.02],[0.74, 0.25],[0.70, 0.20],[0.69, 0.18],[0.74, 0.21],
            [0.60, 0.00],[0.71, -0.01],[0.64, -0.03],[0.65, -0.05],[0.73, 0.17],
            [0.64, 0.08],[0.63, 0.12],[0.75, -0.02],[0.74, 0.12],[0.68, 0.22],
            [0.77, 0.20],[0.75, 0.05],[0.67, 0.13],[0.61, 0.07],[0.66, -0.00],
        ]
        self.train_index_list = list(range(0, 45))
        self.valid_index_list = list(range(45, 50))
        self.validation_index_pointer = 0
        self.test_index_pointer = 0
        # ENTITIES
        self.agent = Panda()
        self.gripper = PandaGripper()
        self.vision_sensor = VisionSensor('Vision_sensor')
        self.cuboid = Shape('Cuboid')
        self.panda_end_point = Dummy('Panda_tip')

    def init_common(self, position_index):
        # INIT SIMULATOR
        self.pr.stop()
        self.pr.start()
        initial_joint_position = [0.021072905510663986, -0.10558620095252991,
                                  0.03912815451622009, -2.3803772926330566,
                                  0.0037031176034361124, 2.22170352935791, 0.8429362773895264]
        ''' MOVE JOINT METHOD 1 '''
        self.agent.set_joint_positions(initial_joint_position, disable_dynamics=True)
        ''' MOVE JOINT METHOD 2
        self.agent.set_joint_target_positions(initial_joint_position)
        [self.pr.step() for _ in range(30)]
        '''
        # TARGET POSITION
        [cub_pos_x, cub_pos_y] = self.cuboid_position_list[position_index]
        self.cuboid.set_position([cub_pos_x, cub_pos_y, 0.82])
        # INIT DYNAMICS
        [self.pr.step() for _ in range(5)]
        # INIT ARM POSITION
        self.move(dim=2, step=-0.02)

    def init_train(self):
        position_index = random.choice(self.train_index_list)
        self.init_common(position_index=position_index)
        self.mode = 'train'
        self.gather_num, self.step_num = 0, 0

    def init_val(self):
        position_index = self.valid_index_list[self.validation_index_pointer]
        self.init_common(position_index=position_index)
        self.validation_index_pointer = (self.validation_index_pointer + 1) % len(self.valid_index_list)
        self.mode = 'val'
        self.gather_num, self.step_num = 0, 0

    def init_test(self):
        position_index = self.valid_index_list[self.test_index_pointer]
        self.init_common(position_index=position_index)
        self.test_index_pointer = (self.test_index_pointer + 1) % len(self.valid_index_list)
        self.mode = 'test'
        self.gather_num, self.step_num = 0, 0

    def get_observation(self):
        obs_image = self.vision_sensor.capture_rgb()
        ''' NOTE: obs_image range [0, 1] '''
        s = self.img_preprocess_with_reshape(obs_image)
        s_tensor = self.state_img_to_tensor(s)
        return s_tensor

    def img_preprocess_with_reshape(self, img):
        img_process = self.img_preprocess(img)
        value_min = 0.40
        value_max = 0.90
        img_process = (np.clip(img_process, a_min=value_min, a_max=value_max) - value_min) / (value_max - value_min)
        ''' CHECK IMAGE
        utils.check_image(img_process, name='1')
        utils.check_image(img, name='2')
        '''
        return img_process.reshape([-1])

    def img_preprocess(self, img):
        '''
            img.shape == (64, 64, 3), maximum value=255. reshape -> 4096
        '''
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.astype(np.float32)
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

    def get_distance(self, pos_1, pos_2):
        d_x = pos_1[0] - pos_2[0]
        d_y = pos_1[1] - pos_2[1]
        d_z = pos_1[2] - pos_2[2]
        distance_value = math.sqrt(d_x**2 + d_y**2 + d_z**2)
        distance_value_2D = math.sqrt(d_x**2 + d_y**2)
        return distance_value
    
    def get_env_variables(self):
        cuboid_pos = self.cuboid.get_position()
        panda_pos = self.panda_end_point.get_position()
        
        target_distance = self.get_distance(pos_1=cuboid_pos, pos_2=panda_pos)
        x_near_flag = abs(panda_pos[0] - cuboid_pos[0]) < 0.03
        y_near_flag = abs(panda_pos[1] - cuboid_pos[1]) < 0.03
        z_near_flag = abs(panda_pos[2] - cuboid_pos[2]) < 0.05
        target_reached = x_near_flag and y_near_flag and z_near_flag

        return {'target_distance': target_distance,
                'target_in_view': target_reached,
                'cuboid_pos': cuboid_pos,
                'panda_pos': panda_pos,
            }
    
    def move(self, dim=0, step=0.1):
        (x, y, z), q = self.agent.get_tip().get_position(), self.agent.get_tip().get_quaternion()
        if dim == 0:
            x += step
        elif dim == 1:
            y += step
        elif dim == 2:
            z += step
        try:
            new_joint_pos = self.agent.solve_ik_via_jacobian([x, y, z], quaternion=q)
        except:
            print('IKError')
        else:
            self.agent.set_joint_positions(new_joint_pos, disable_dynamics=True)

    def make_action(self, action):
        # -----old---------------
        old_env_variables = self.get_env_variables()
        old_distance = old_env_variables['target_distance']
        old_target_in_view = old_env_variables['target_in_view']
        # -----make action-------
        action_index = torch.argmax(action).item()
        action_string = self.ACTIONS[action_index]
        action_success = True
        if action_string == 0:
            if old_env_variables['panda_pos'][0] > 0.80 + 0.05:
                action_success = False
            else:
                self.move(dim=0, step=0.03)
        elif action_string == 1:
            if old_env_variables['panda_pos'][0] < 0.60 - 0.05:
                action_success = False
            else:
                self.move(dim=0, step=-0.03)
        elif action_string == 2:
            if old_env_variables['panda_pos'][1] > 0.25 + 0.05:
                action_success = False
            else:
                self.move(dim=1, step=0.03)
        elif action_string == 3:
            if old_env_variables['panda_pos'][1] < -0.05 - 0.05:
                action_success = False
            else:
                self.move(dim=1, step=-0.03)
        elif action_string == 9:
            if old_env_variables['panda_pos'][2] < 0.775 + 0.04:        # 0.03
                ''' BOUNDARY, STOP'''
                action_success = False
            else:
                self.move(dim=2, step=-0.05)
        # elif action_string == 8:
        #     if old_env_variables['panda_pos'][2] > 0.775 + 0.25:
        #         ''' BOUNDARY, STOP'''
        #         action_success = False
        #     else:
        #         self.move(dim=2, step=+0.05)
        # -----new state---------
        new_env_variables = self.get_env_variables()
        new_distance = new_env_variables['target_distance']
        new_target_in_view = new_env_variables['target_in_view']
        # -----reward------------
        if new_target_in_view == True:
            done_flag = True
            r = 10
        else:
            done_flag = False
            r = 0
            # if action_success == False:
            #     r = -5
            if new_distance < old_distance:
                r = +1
                self.gather_num += 1
            # elif new_distance > old_distance:
            #     r = -1
            # else:
            #     r = -1
        self.step_num += 1
        # -----output------------
        self.done_signal = done_flag
        reward = torch.Tensor([r]).to(self.dev)
        ''' NOTE: self.gather_num is the number of approaching steps'''
        ''' NOTE: variable[0] step num is checked to save for larger values'''
        # CHECK
        if self.SHOW_STATE_VARIABLES == True:
            print('>  ', reward, self.step_num, self.done_signal)
            print('     ', new_env_variables['panda_pos'])
            print('     ', new_env_variables['cuboid_pos'])
            print('     ', old_distance, new_distance, new_target_in_view)
        if self.SAVE_STATE_IMAGE == True:
            utils.check_image(self.vision_sensor.capture_rgb())
        
        return reward, [-self.step_num, self.gather_num]




if __name__ == "__main__":
    pass
    


print('\033[91mFINISH: env_robotarm\033[0m')

