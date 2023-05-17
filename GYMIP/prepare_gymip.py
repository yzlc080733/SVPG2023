# -*- coding: utf-8 -*-
import numpy as np
import xml.etree.ElementTree as Et
import time
# import importlib.util
import sys
import os
sys.path.insert(0, os.getcwd() + '/env')
import gym

# pip install gym[mujoco]. As of 2022, this should automatically install a pre-packaged mujoco lib.
# 'inverted_pendulum.xml' copied from gym.

# # Load gym library from local folder. Modified InvertedPendulum to support extra XML files.
# spec = importlib.util.spec_from_file_location("gfg", "./env/gym_lib_custom/__init__.py")
# gym_local = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(gym_local)


default_length = 1.5
default_thickness = 0.05


def print_info(input_string):
    print('\033[93mGYMIP_PRE_INFO|\033[0m', input_string)


def edit_xml_file(filename_1, filename_2, length, thickness):
    tree = Et.parse(filename_1)
    root = tree.getroot()
    root[4][1][2][1].attrib['fromto'] = '0 0 0 0.001 0 %f' % length
    root[4][1][2][1].attrib['size'] = '%f 0.3' % thickness
    tree.write(filename_2)


def prepare_xml_files(path='./env/gym/envs/mujoco/assets/', base_file_name='inverted_pendulum.xml'):
    fullname = path + base_file_name
    print_info(fullname)
    # Change length
    for noise_param in np.arange(0.16, 4.88, 0.08):
        filename_output = path + 'inverted_pendulum_ChangeLen_%f' % (noise_param) + '.xml'
        edit_xml_file(fullname, filename_output, length=noise_param, thickness=default_thickness)
    # Change thickness
    for noise_param in np.arange(0.01, 0.305, 0.005):
        filename_output = path + 'inverted_pendulum_ChangeThk_%f' % (noise_param) + '.xml'
        edit_xml_file(fullname, filename_output, length=default_length, thickness=noise_param)
    # Change length and thickness simultaneously
    for noise_param in np.arange(0.02, 0.305, 0.005):
        filename_output = path + 'inverted_pendulum_ChangeBoth_%f' % (noise_param) + '.xml'
        edit_xml_file(fullname, filename_output, length=noise_param * 16, thickness=noise_param)


if __name__ == "__main__":
    prepare_xml_files()

    DISPLAY_ENVS = False
    if DISPLAY_ENVS:
        for noise_param in np.arange(0.16, 4.88, 0.08):
            file_name_try = 'inverted_pendulum_ChangeLen_%f' % (noise_param) + '.xml'
        # for noise_param in np.arange(0.01, 0.305, 0.005):
        #     file_name_try = 'inverted_pendulum_ChangeThk_%f' % (noise_param) + '.xml'
        # for noise_param in np.arange(0.02, 0.305, 0.005):
        #     file_name_try = 'inverted_pendulum_ChangeBoth_%f' % (noise_param) + '.xml'

            env = gym.make('InvertedPendulum-v4', render_mode='human', self_xml=file_name_try)
            print(env.model.body_mass)
            for episode_i in range(2):
                env.reset()
                for step_i in range(30):
                    if int(step_i / 5) % 4 in [3, 0]:
                        action = 0.2
                    else:
                        action = -0.2
                    env.step([action])
                    env.render()
                    time.sleep(0.02)
            env.close()

    print('\033[91mFINISH: prepare_gymip\033[0m')



