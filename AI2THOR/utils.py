# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
import pickle
import cv2
import torch
import math

'''  UTILS  '''


def image_resize_preprocess(image_input, SIZE_X, SIZE_Y):
    # Gray Background
    output_image = np.ones([SIZE_Y, SIZE_X, 3], dtype=np.uint8) * 128
    ratio_0 = SIZE_Y / image_input.shape[0]
    ratio_1 = SIZE_X / image_input.shape[1]
    # Ratio for Resize
    if ratio_0 < ratio_1:
        re_size_0 = SIZE_Y
        re_size_1 = math.floor(image_input.shape[1] * ratio_0)
        input_image_resize = cv2.resize(image_input, (re_size_1, re_size_0), interpolation=cv2.INTER_AREA)
    else:
        re_size_0 = math.floor(image_input.shape[0] * ratio_1)
        re_size_1 = SIZE_X
        input_image_resize = cv2.resize(image_input, (re_size_1, re_size_0), interpolation=cv2.INTER_AREA)
    # Position of the Image
    pos0_1 = math.floor(SIZE_Y*0.5 - input_image_resize.shape[0] / 2)
    pos0_2 = pos0_1 + input_image_resize.shape[0]
    pos1_1 = math.floor(SIZE_X*0.5 - input_image_resize.shape[1] / 2)
    pos1_2 = pos1_1 + input_image_resize.shape[1]
    output_image[pos0_1:pos0_2, pos1_1:pos1_2, :] = input_image_resize
    return output_image


def print_info(input_string):
    print('\033[94mINFO|\033[0m', input_string)


def save_file(file_path, file_content):
    with open(file_path, 'wb') as f:
        pickle.dump(file_content, f)


def clip_value(x, min_x, max_x):
    return min(max(x, min_x), max_x)


def index_to_onehot(index, total_number):
    output_vector = np.zeros(total_number, dtype=np.float32)
    output_vector[index] = 1
    return output_vector


def check_image(img, name=''):
    img_write = None
    dimension = len(img.shape)
    if type(img) == torch.Tensor:
        img = img.cpu().numpy()
    print_info('MAX MIN: ' + str(img.max()) + '   ' + str(img.min()))
    if dimension > 4 or dimension < 2:
        pass
    elif dimension == 2:
        img_write = img
    else:       # dimension == 3
        if img.shape[0] == 3:
            img_write = np.moveaxis(img, 0, 2)
            img_write = cv2.cvtColor(img_write, cv2.COLOR_RGB2BGR)
        elif img.shape[2] == 3:
            img_write = img
    if img_write is None:
        print_info('error in img shape: ' + str(img.shape))
    else:
        if not os.path.exists('./temp_check/'):
            os.mkdir('./temp_check/')
        cv2.imwrite('./temp_check/chk_' + name + '.png', img_write / img_write.max() * 255)
        print_info('img shape: ' + str(img.shape))


def pause_to_check():
    print_info('ready to check')
    sys.exit(0)
    time.sleep(1000)


def colored_print(string, color:int=1):
    print('\033[9%dm%s\033[0m' % (color, str(string)))
    '''
        RED     = '\033[31m'
        GREEN   = '\033[32m'
        YELLOW  = '\033[33m'
        BLUE    = '\033[34m'
        MAGENTA = '\033[35m'
        CYAN    = '\033[36m'
        WHITE   = '\033[37m'
        RESET   = '\033[39m'
    '''


def inspect_dict(dict_input):
    for key in dict_input.keys():
        print(key, type(dict_input[key]))



def get_exp_name_from_args(args, initial_string=''):
    string = initial_string
    for arg_key in vars(args).keys():
        if not arg_key in ['cuda', 'thread']:
            string = string + '_' + str(vars(args)[arg_key])
    return string


# print_info('IMPORT UTILS')
