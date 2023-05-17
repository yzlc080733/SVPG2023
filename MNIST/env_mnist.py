# -*- coding: utf-8 -*-

import math
import numpy as np
import torch
import random
import skimage
import prepare_mnist



# ENV mnist


def print_mnist_info(input_string):
    MNIST_ENV_INFO_MESSAGE = '\033[92mMNIST_ENV_INFO|\033[0m'
    print(MNIST_ENV_INFO_MESSAGE, input_string)


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


def convert_data(input_variable, output_type_string='cpu', gpu_device=None):
    input_variable_type = check_data_type(input_variable)
    if not output_type_string in ['cpu', 'torch_cpu', 'torch_gpu']:
        print_mnist_info('Error in target data type.')
        return None
    if input_variable_type == None:
        print_mnist_info('Error in input_variable type.')
        return None
    else:
        if input_variable_type == 'cpu':
            if output_type_string == 'cpu':
                return np.copy(input_variable)
            elif output_type_string == 'torch_cpu':
                return torch.from_numpy(np.copy(input_variable))
            elif output_type_string == 'torch_gpu':
                return torch.from_numpy(np.copy(input_variable)).to(gpu_device)
        elif input_variable_type == 'torch_cpu':
            if output_type_string == 'cpu':
                return input_variable.detach().numpy()
            elif output_type_string == 'torch_cpu':
                return input_variable
            elif output_type_string == 'torch_gpu':
                return input_variable.to(gpu_device)
        elif input_variable_type == 'torch_gpu':
            if output_type_string == 'cpu':
                return input_variable.detach().cpu().numpy()
            elif output_type_string == 'torch_cpu':
                return input_variable.detach().cpu()
            elif output_type_string == 'torch_gpu':
                return input_variable


def get_onehot_from_numpy(array, num_class):
    if len(array.shape) != 1:
        print_mnist_info('Error in array shape (onehot conversion)')
        return None
    else:
        output_onehot = np.zeros([array.shape[0], num_class], dtype=np.float32)
        output_onehot[np.arange(array.shape[0]), array] = 1
        return output_onehot




class MNIST_DATASET:
    def __init__(self):
        mnist_content = prepare_mnist.load_mnist()
        self.train_image_org = mnist_content['training_images']     # uint8 [0, 255]
        self.train_label_org = mnist_content['training_labels']
        self.test_image_org = mnist_content['test_images']
        self.test_label_org = mnist_content['test_labels']
        self.train_image_org = self.train_image_org.astype(np.float32) / 255
        self.test_image_org = self.test_image_org.astype(np.float32) / 255

        self.dataset_split()

        self.train_label_onehot = get_onehot_from_numpy(self.train_label, 10)
        self.val_label_onehot = get_onehot_from_numpy(self.val_label, 10)
        self.test_label_onehot = get_onehot_from_numpy(self.test_label, 10)
        
        self.data_position = 'cpu'

        self.test_get_flag, self.val_get_flag = 0, 0

        self.max_step_num = 100000
        self.state_dimension = 784
        self.action_num = 10

    def dataset_split(self, train_split_ratio=0.9):
        DATA_NUM = self.train_image_org.shape[0]
        random_index_list = np.random.permutation(DATA_NUM)
        train_data_num = math.floor(DATA_NUM * train_split_ratio)
        val_data_num = DATA_NUM - train_data_num                # train 0.9 | val 0.1
        
        train_index_list = random_index_list[0:train_data_num]
        val_index_list = random_index_list[train_data_num:(train_data_num + val_data_num)]
        
        self.train_image = self.train_image_org[train_index_list]
        self.train_label = self.train_label_org[train_index_list]
        self.val_image = self.train_image_org[val_index_list]
        self.val_label = self.train_label_org[val_index_list]
        self.test_image = self.test_image_org
        self.test_label = self.test_label_org

        print_mnist_info('TRAIN %d | VAL %d | TEST %d' % (self.train_image.shape[0],
                self.val_image.shape[0], self.test_image.shape[0]))

    def convert_all(self, target_device_string, gpu_device=None):
        self.train_image = convert_data(self.train_image, target_device_string, gpu_device)
        self.train_label = convert_data(self.train_label, target_device_string, gpu_device)
        self.train_label_onehot = convert_data(self.train_label_onehot, target_device_string, gpu_device)

        self.val_image = convert_data(self.val_image, target_device_string, gpu_device)
        self.val_label = convert_data(self.val_label, target_device_string, gpu_device)
        self.val_label_onehot = convert_data(self.val_label_onehot, target_device_string, gpu_device)
        
        self.test_image = convert_data(self.test_image, target_device_string, gpu_device)
        self.test_label = convert_data(self.test_label, target_device_string, gpu_device)
        self.test_label_onehot = convert_data(self.test_label_onehot, target_device_string, gpu_device)
        
        self.data_position = target_device_string

    def convert_all_to_torch_cpu(self):
        self.convert_all('torch_cpu')

    def convert_all_to_torch_gpu(self):
        self.convert_all('torch_gpu', torch.device('cuda:0'))

    # ~~~~~~~~~~~~~~~~~~~~TRAIN~~~~~~~~~~~~~~~~~~~~~~~~~
    def init_train(self):
        self.done_signal = False
        self.mode = 'train'
        self.total_num, self.accurate_num = 0, 0

    def init_val(self):
        self.done_signal = False
        self.mode = 'val'
        self.total_num, self.accurate_num = 0, 0
        self.get_batch_flag = 0

    def init_test(self):
        self.done_signal = False
        self.mode = 'test'
        self.total_num, self.accurate_num = 0, 0
        self.get_batch_flag = 0

    def get_train_observation(self, batch_size=100, **kwargs):
        if batch_size > self.train_image.shape[0]:
            print_mnist_info('Error batch size too large: ', batch_size, self.train_image.shape[0])
            return None
        else:
            batch_list = random.sample(range(self.train_image.shape[0]), batch_size)
            batch_train_image = self.train_image[batch_list]
            batch_train_label = self.train_label[batch_list]
            batch_train_label_onehot = self.train_label_onehot[batch_list]
            # -----prepare reward----
            self.calc_reward_label = batch_train_label
            self.calc_reward_label_onehot = batch_train_label_onehot
            return batch_train_image
    
    def make_action(self, action):
        # -----done signal-------
        if self.mode == 'train':
            self.done_signal = True
        else:
            if self.get_batch_end == True:
                self.done_signal = True
            else:
                self.done_signal = False
        # -----accuracy----------
        accurate_item = torch.sum(torch.abs(action - self.calc_reward_label_onehot), dim=1) == 0
        accurate_num = torch.sum(accurate_item)
        self.accurate_num = self.accurate_num + accurate_num
        self.total_num = self.total_num + action.shape[0]
        accuracy = self.accurate_num / self.total_num
        # -----reward------------
        reward = accurate_item.float() * 2 - 1                      # [-1, +1]
        return reward, [accuracy]

    def get_val_observation(self, batch_size=100, **kwargs):
        if batch_size > self.val_image.shape[0]:
            print_mnist_info('Error batch size too large: ', batch_size, self.val_image.shape[0])
            return None
        else:
            start_point = self.get_batch_flag * batch_size
            end_point = (self.get_batch_flag + 1) * batch_size
            if start_point >= self.val_image.shape[0]:
                print_mnist_info('Error index exceeds boundary')
                return None
            else:
                if end_point >= self.val_image.shape[0]:
                    self.get_batch_end = True
                    end_point = self.val_image.shape[0]
                else:
                    self.get_batch_end = False
                batch_val_image = self.val_image[start_point:end_point]
                batch_val_label = self.val_label[start_point:end_point]
                batch_val_label_onehot = self.val_label_onehot[start_point:end_point]
                self.get_batch_flag = self.get_batch_flag + 1
                # -----prepare reward----
                self.calc_reward_label = batch_val_label
                self.calc_reward_label_onehot = batch_val_label_onehot
                return batch_val_image

    def get_test_observation(self, batch_size=100, noise_type='none', noise_param=0, **kwargs):
        self.add_noise_to_test_image_buffer(noise_type, noise_param)
        if batch_size > self.test_image.shape[0]:
            print_mnist_info('Error batch size too large: ', batch_size, self.test_image.shape[0])
            return None
        else:
            start_point = self.get_batch_flag * batch_size
            end_point = (self.get_batch_flag + 1) * batch_size
            if start_point >= self.test_image.shape[0]:
                print_mnist_info('Error index exceeds boundary')
                return None
            else:
                if end_point >= self.test_image.shape[0]:
                    self.get_batch_end = True
                    end_point = self.test_image.shape[0]
                else:
                    self.get_batch_end = False
                batch_test_image = self.test_image_noise[start_point:end_point]     # with noise
                batch_test_label = self.test_label[start_point:end_point]
                batch_test_label_onehot = self.test_label_onehot[start_point:end_point]
                self.get_batch_flag = self.get_batch_flag + 1
                # -----prepare reward----
                self.calc_reward_label = batch_test_label
                self.calc_reward_label_onehot = batch_test_label_onehot
                return batch_test_image
    
    def add_noise_to_test_image_buffer(self, noise_type, noise_param):
        if noise_type == 'none':
            self.test_image_noise = torch.clone(self.test_image)
            return
        self.test_image_noise = np.copy(convert_data(self.test_image, 'cpu'))
        for img_i in range(self.test_image_noise.shape[0]):
            temp_img = self.test_image_noise[img_i, :].reshape([28, 28])
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
            self.test_image_noise[img_i, :] = temp_img.reshape([784])
        self.test_image_noise = convert_data(self.test_image_noise, self.data_position, torch.device('cuda:0'))



if __name__ == "__main__":
    mnist = MNIST_DATASET()
    mnist.convert_all_to_torch_cpu()
    mnist.convert_all_to_torch_gpu()
    # -----train-------------
    mnist.init_train()
    A = mnist.get_train_observation()
    B = torch.zeros([A.shape[0], mnist.action_num]).cuda()
    mnist.make_action(B)
    print(mnist.done_signal)
    # -----validate----------
    mnist.init_val()
    while True:
        A = mnist.get_val_observation(batch_size=1002)
        B = torch.zeros([A.shape[0], mnist.action_num]).cuda()
        mnist.make_action(B)
        print('\033[92m>>>\033[0m', A.shape[0])
        if mnist.done_signal == True:
            break
    # -----test--------------
    mnist.init_test()
    while True:
        A = mnist.get_test_observation(batch_size=800)
        B = torch.zeros([A.shape[0], mnist.action_num]).cuda()
        mnist.make_action(B)
        print('\033[92m>>>\033[0m', A.shape[0])
        if mnist.done_signal == True:
            break
    


print('\033[91mFINISH: env_mnist\033[0m')

