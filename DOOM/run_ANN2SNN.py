# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch
import argparse
import datetime
import re
import time
import itertools
import copy

import torch
import torch.nn as nn

import memory_lib


# PPO version
# python run_RL.py --task mnist --model mlp3relu --seed 6 --ignore_checkpoint --optimizer sgd --lr 0.0001



def get_arguments():
    parser = argparse.ArgumentParser(description='Description: run_RL')
    # -----RL general--------
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.97)
    parser.add_argument('--entropy', type=float, default=0.02)
    parser.add_argument('--target_network_freq', type=int, default=100)
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--PPO_epochs', type=int, default=10)
    parser.add_argument('--eps_clip', type=float, default=0.2)

    parser.add_argument('--task', type=str, default='vizdoom',
            choices=['mnist', 'cifar10', 'vizdoom'])
    parser.add_argument('--model', type=str, default='mlp3relu',
            choices=['mlp3relu'])
    parser.add_argument('--optimizer', type=str, default='adam',
            choices=['sgd', 'adam', 'rmsprop'])
    # ---------------------------------------------------
    # for mlp3, snnbptt
    parser.add_argument('--hidden_num', type=int, default=500)
    # for rwta
    parser.add_argument('--hid_group_num', type=int, default=50)
    parser.add_argument('--hid_group_size', type=int, default=10)
    parser.add_argument('--rwta_del_connection', type=str, default='none',
            choices=['none', 'hh', 'sa', 'hhsa'])
    # for snnbptt
    parser.add_argument('--snn_num_steps', type=int, default=25)
    # ---------------------------------------------------
    parser.add_argument('--train_num', type=int, default=20000)
    parser.add_argument('--val_freq', type=int, default=100)
    parser.add_argument('--val_num', type=int, default=10)
    parser.add_argument('--test_num', type=int, default=10)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--ignore_checkpoint', default=False, action='store_true')
    return parser.parse_args()


def reload_log_file(filename):
    train_epi_num, val_best = None, None
    with open(filename) as file:
        line_index = 0
        for line in file:
            str_list = [i for i in re.sub(',', ' ', line).split()]
            if str_list[0] == 'train':
                train_epi_num = int(str_list[1])
            if str_list[0] == 'val':
                val_best = float(str_list[2])
    if train_epi_num == None:
        train_epi_num = 0
        val_best = -10000
    if val_best == None:
        val_best = -10000
    return train_epi_num, val_best


def log_text(file_handle, type_str, record_text, onscreen=True):
    global log_text_flush_time
    if onscreen:
        print('\033[92m%s\033[0m' % (type_str).ljust(10), record_text)
    file_handle.write((type_str+',').ljust(10) + record_text + '\n')
    if time.time() - log_text_flush_time > 10:
        log_text_flush_time = time.time()
        file_handle.flush()
        os.fsync(file_handle.fileno())





if __name__ == "__main__":
    # -----arguments---------
    args = get_arguments()
    if args.model in ['mlp3relu',]:
        model_hyperparam_str = 'h%d' % (args.hidden_num)
    else:
        print('\033[91mError in arguments\033[0m')
    EXP_NAME = 'PPO_%s_%s_%s_%s_%8.6f_%4.2f_%6.5f_%d_%5.4f_rep%02d' % (
            args.task, args.model,
            model_hyperparam_str,
            args.optimizer, args.lr, args.entropy, args.gamma,
            args.PPO_epochs, args.eps_clip,
            args.seed)
    # -----cuda device-------
    if args.cuda < 0:
        torch_device = torch.device('cpu')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '%1d' % args.cuda
        torch_device = torch.device('cuda:0')
    # -----env---------------
    if args.task == 'mnist':
        import env_mnist
        env = env_mnist.MNIST_DATASET()
        if args.cuda < 0:
            env.convert_all_to_torch_cpu()
        else:
            env.convert_all_to_torch_gpu()
        input_dimension, output_dimension = env.state_dimension, env.action_num
        train_batch_size, test_batch_size = 100, 10000
        mem = memory_lib.MemoryBatch()
    elif args.task == 'cifar10':
        import env_cifar10
        env = env_cifar10.CIDAR10_DATASET()
        if args.cuda < 0:
            env.convert_all_to_torch_cpu()
        else:
            env.convert_all_to_torch_gpu()
        input_dimension, output_dimension = env.state_dimension, env.action_num
        train_batch_size, test_batch_size = 100, 10000
        mem = memory_lib.MemoryBatch()
    elif args.task == 'vizdoom':
        import env_vizdoom
        env = env_vizdoom.VIZDOOM_HEALTHGATHERING()
        if args.cuda < 0:
            env.convert_all_to_torch_cpu()
        else:
            env.convert_all_to_torch_gpu()
        input_dimension, output_dimension = env.state_dimension, env.action_num
        train_batch_size, test_batch_size = 1, 1
        mem = memory_lib.MemoryList(gamma=args.gamma)
    # -----model-------------
    if args.model == 'mlp3relu':
        import model_mlp
        model = model_mlp.MLP_3(
                layer_sizes=[input_dimension, args.hidden_num, output_dimension],
                hid_activate='relu', hid_group_size=args.hid_group_size,
                out_activate='none', optimizer_name=args.optimizer, optimizer_learning_rate=args.lr,
                entropy_ratio=args.entropy,)
        # model = model_mlp.MLP_3(
        #         layer_sizes=[input_dimension, args.hidden_num, output_dimension],
        #         hid_activate='relu', hid_group_size=args.hid_group_size,
        #         out_activate='softmax', optimizer_name=args.optimizer, optimizer_learning_rate=args.lr,
        #         entropy_ratio=args.entropy,)

    # -----init timer--------
    log_text_flush_time = time.time()
    model.load_model(EXP_NAME + '_best')
    # -----log file----------
    log_filename = './ANN2SNN/log_text/log_' + EXP_NAME + '.txt'
    File = open(log_filename, 'w')
    log_text(File, 'init_ANN2SNN', str(datetime.datetime.now()))
    log_text(File, 'arguments', str(args))

    model = model.to(torch_device)

    # ~~~~~~~~~~~~~~~~~~~~CONVERT~~~~~~~~~~~~~~~~~~~~~~~
    from spikingjelly.clock_driven.ann2snn import parser, classify_simulator
    from spikingjelly.clock_driven import functional
    print('\033[91m CONVERT\033[0m')
    model_name = 'model_convert_rep%s' % (EXP_NAME)
    log_dir_temp = './ANN2SNN/temp_log_dir_%s/' % (EXP_NAME)
    parser_device = torch_device
    simulator_device = parser_device
    # -----generate data-----
    norm_data_list = []
    while True:
        env.init_train()
        for collect_step_i in range(env.max_step_num):
            observation = env.get_train_observation(batch_size=train_batch_size)
            model_output, model_other_output = model(observation)
            action_chosen_index = torch.argmax(model_output, dim=1)
            action_chosen_onehot = torch.nn.functional.one_hot(action_chosen_index, num_classes=env.action_num)
            reward, _ = env.make_action(action_chosen_onehot)
            norm_data_list.append(observation)
            if env.done_signal == True:
                break
        print(len(norm_data_list))
        if len(norm_data_list) > 20000:
            break
    # -----convert-----------
    norm_data = torch.cat(norm_data_list)
    print('use %d states to parse' % (norm_data.size(0)))

    onnxparser = parser(name=model_name, kernel='onnx', log_dir=log_dir_temp)
    snn = onnxparser.parse(model, norm_data.to(parser_device))

    torch.save(snn, os.path.join(log_dir_temp, 'snn_' + model_name + '.pkl'))

    del norm_data_list, norm_data
    
    # ~~~~~~~~~~~~~~~~~~~~TEST~~~~~~~~~~~~~~~~~~~~~~~~~~
    log_text(File, 'test', str(datetime.datetime.now()))
    # # ~~~~~~~~~~~~~~~~~~~~TEST ADVERSARIAL~~~~~~~~~~~~~~
    # import model_adversarial
    # ad_mem_size = 1000
    # if args.task in ['mnist', 'cifar10']:
    #     ad_train_epi_num = 500
    # else:               # 'vizdoom'
    #     ad_train_epi_num = 100 ###################################################################
    # ad_mem_s = torch.zeros([ad_mem_size, input_dimension]).to(torch_device)
    # ad_mem_a = torch.zeros([ad_mem_size, output_dimension]).to(torch_device)
    # ad_model = model_adversarial.Adversarial(input_dimension, output_dimension)
    # ad_model = ad_model.to(torch_device)
    # pointer, total_num = 0, 0

    # # -----load--------------
    # snn = torch.load(os.path.join(log_dir_temp, 'snn_' + model_name + '.pkl')).to(torch_device)
    # sim = classify_simulator(snn, log_dir=log_dir_temp + 'simulator', device=simulator_device)

    # for collect_epi_i in range(ad_train_epi_num):
    #     env.init_train()
    #     for collect_step_i in range(env.max_step_num):
    #         observation = env.get_train_observation(batch_size=train_batch_size)
            
    #         # -----spike-------------
    #         with torch.no_grad():
    #             functional.reset_net(snn)
    #             for snn_t in range(args.snn_num_steps):
    #                 enc = sim.encoder(observation).float()
    #                 out = snn(enc)
    #                 if snn_t == 0:
    #                     counter = out
    #                 else:
    #                     counter = counter + out
    #         model_output = counter

    #         action_chosen_index = torch.argmax(model_output, dim=1)
    #         action_chosen_onehot = torch.nn.functional.one_hot(action_chosen_index, num_classes=env.action_num)
    #         reward, _ = env.make_action(action_chosen_onehot)
    #         for sample_i in range(observation.shape[0]):
    #             ad_mem_s[pointer, :] = observation[sample_i, :]
    #             ad_mem_a[pointer, :] = action_chosen_onehot[sample_i, :]
    #             pointer = (pointer + 1) % ad_mem_size
    #             total_num = min(ad_mem_size, total_num + 1)
    #         if env.done_signal == True:
    #             break
    #     sample_list = random.sample(range(0, total_num), min(total_num, 200))
    #     ad_s_batch = ad_mem_s[sample_list]
    #     ad_a_batch = ad_mem_a[sample_list]
    #     ad_predict = ad_model(ad_s_batch)
    #     ad_loss = ad_model.learn(ad_s_batch, ad_predict, ad_a_batch)
    #     log_text(File, 'ADtrain', '%8d,   %8.6f' % (collect_epi_i, ad_loss))
    # perturb_loss = nn.CrossEntropyLoss()
    
    # for epsilon in np.arange(0, 0.2, 0.01):
    #     test_preformance_list = []
    #     for test_epi_i in range(args.test_num):
    #         env.init_test()
    #         for test_step_i in range(env.max_step_num):
    #             observation = env.get_test_observation(batch_size=test_batch_size)
    #             observation.requires_grad = True            # for FGSM
    #             ad_output = ad_model(observation)
    #             ad_argmax = torch.argmax(ad_output, dim=1)
    #             perturb_loss_value = perturb_loss(ad_output, ad_argmax).mean()
    #             ad_model.zero_grad()
    #             perturb_loss_value.backward()
    #             observation_grad = observation.grad.data
    #             sign_grad = observation_grad.sign()
    #             observation_perturb = observation + epsilon * sign_grad

    #             # -----spike-------------
    #             with torch.no_grad():
    #                 functional.reset_net(snn)
    #                 for snn_t in range(args.snn_num_steps):
    #                     enc = sim.encoder(observation_perturb).float()
    #                     out = snn(enc)
    #                     if snn_t == 0:
    #                         counter = out
    #                     else:
    #                         counter = counter + out
    #             model_output = counter
                
    #             action_chosen_index = torch.argmax(model_output, dim=1)
    #             action_chosen_onehot = torch.nn.functional.one_hot(action_chosen_index, num_classes=env.action_num)
    #             reward, test_other_step_record = env.make_action(action_chosen_onehot)
    #             if env.done_signal == True:
    #                 break
    #         test_preformance_list.append(test_other_step_record[0])
    #     test_performance_mean = sum(test_preformance_list) / len(test_preformance_list)
    #     log_text(File, 'FGSM', '%8.6f,   %8.6f' % (epsilon, test_performance_mean))
    #     File.flush()


    # >>>> Test Weight REL
    noise_type_list = ['gaussian', 'uniform']
    for noise_type in noise_type_list:
        # -----noise param-------
        if noise_type in ['gaussian']:
            noise_param_list = np.arange(0, 5, 0.1)
        else:    # if noise_type in ['uniform']:
            noise_param_list = np.arange(0, 5, 0.1)
        for noise_param in noise_param_list:
            test_preformance_list = []
            for test_epi_i in range(args.test_num):
                # -----load--------------
                snn = torch.load(os.path.join(log_dir_temp, 'snn_' + model_name + '.pkl')).to(torch_device)
                with torch.no_grad():
                    for param in snn.parameters():
                        mean_value = np.mean(np.abs(param.cpu().numpy()))
                        if noise_type == 'gaussian':
                            param.add_(torch.randn(param.size()).to(torch_device) * noise_param * mean_value)
                        if noise_type == 'uniform':
                            param.add_((torch.rand(param.size()).to(torch_device) - 0.5) * 2 * noise_param * mean_value)
                sim = classify_simulator(snn, log_dir=log_dir_temp + 'simulator', device=simulator_device)
                
                env.init_test()
                for test_step_i in range(env.max_step_num):
                    observation = env.get_test_observation(batch_size=test_batch_size)

                    # -----spike-------------
                    with torch.no_grad():
                        functional.reset_net(snn)
                        for snn_t in range(args.snn_num_steps):
                            enc = sim.encoder(observation).float()
                            out = snn(enc)
                            if snn_t == 0:
                                counter = out
                            else:
                                counter = counter + out
                    model_output = counter

                    action_chosen_index = torch.argmax(model_output, dim=1)
                    action_chosen_onehot = torch.nn.functional.one_hot(action_chosen_index, num_classes=env.action_num)
                    reward, test_other_step_record = env.make_action(action_chosen_onehot)
                    if env.done_signal == True:
                        break
                test_preformance_list.append(test_other_step_record[0])
            test_performance_mean = sum(test_preformance_list) / len(test_preformance_list)
            log_text(File, 'w_rel_noise', '%s,  %8.6f,   %8.6f' % (noise_type, noise_param, test_performance_mean))
            File.flush()


    # -----test weight abs---
    noise_type_list = ['gaussian', 'uniform']
    for noise_type in noise_type_list:
        # -----noise param-------
        if noise_type in ['gaussian']:
            noise_param_list = np.arange(0, 1.0, 0.02)
        else:    # if noise_type in ['uniform']:
            noise_param_list = np.arange(0, 4.0, 0.05)
        for noise_param in noise_param_list:
            test_preformance_list = []
            for test_epi_i in range(args.test_num):
                # -----load--------------
                snn = torch.load(os.path.join(log_dir_temp, 'snn_' + model_name + '.pkl')).to(torch_device)
                with torch.no_grad():
                    for param in snn.parameters():
                        if noise_type == 'gaussian':
                            param.add_(torch.randn(param.size()).cuda() * noise_param)
                        if noise_type == 'uniform':
                            param.add_((torch.rand(param.size()).cuda() - 0.5) * 2 * noise_param)
                sim = classify_simulator(snn, log_dir=log_dir_temp + 'simulator', device=simulator_device)
                
                env.init_test()
                for test_step_i in range(env.max_step_num):
                    observation = env.get_test_observation(batch_size=test_batch_size)

                    # -----spike-------------
                    with torch.no_grad():
                        functional.reset_net(snn)
                        for snn_t in range(args.snn_num_steps):
                            enc = sim.encoder(observation).float()
                            out = snn(enc)
                            if snn_t == 0:
                                counter = out
                            else:
                                counter = counter + out
                    model_output = counter

                    action_chosen_index = torch.argmax(model_output, dim=1)
                    action_chosen_onehot = torch.nn.functional.one_hot(action_chosen_index, num_classes=env.action_num)
                    reward, test_other_step_record = env.make_action(action_chosen_onehot)
                    if env.done_signal == True:
                        break
                test_preformance_list.append(test_other_step_record[0])
            test_performance_mean = sum(test_preformance_list) / len(test_preformance_list)
            log_text(File, 'w_noise', '%s,  %8.6f,   %8.6f' % (noise_type, noise_param, test_performance_mean))
            File.flush()
    


    # -----test input--------
    snn = torch.load(os.path.join(log_dir_temp, 'snn_' + model_name + '.pkl')).to(torch_device)
    sim = classify_simulator(snn, log_dir=log_dir_temp + 'simulator', device=simulator_device)

    noise_type_list = ['gaussian', 'pepper', 'salt', 's&p', 'gaussian&salt']
    for noise_type in noise_type_list:
        if noise_type in ['gaussian']:
            noise_param_list = np.arange(0, 1.6, 0.05)
        if noise_type in ['pepper', 'salt', 's&p']:
            noise_param_list = np.arange(0, 0.51, 0.02)
        if noise_type in ['gaussian&salt']:
            noise_param_list = np.arange(0, 0.505, 0.005)
        for noise_param in noise_param_list:
            test_preformance_list = []
            for test_epi_i in range(args.test_num):
                env.init_test()
                for test_step_i in range(env.max_step_num):
                    observation = env.get_test_observation(batch_size=test_batch_size,
                            noise_type=noise_type, noise_param=noise_param)
                    
                    # -----spike-------------
                    with torch.no_grad():
                        functional.reset_net(snn)
                        for snn_t in range(args.snn_num_steps):
                            enc = sim.encoder(observation).float()
                            out = snn(enc)
                            if snn_t == 0:
                                counter = out
                            else:
                                counter = counter + out
                    model_output = counter

                    action_chosen_index = torch.argmax(model_output, dim=1)
                    action_chosen_onehot = torch.nn.functional.one_hot(action_chosen_index, num_classes=env.action_num)
                    reward, test_other_step_record = env.make_action(action_chosen_onehot)

                    if env.done_signal == True:
                        break
                test_preformance_list.append(test_other_step_record[0])
            test_performance_mean = sum(test_preformance_list) / len(test_preformance_list)
            log_text(File, 'i_noise', '%s,  %8.6f,   %8.6f' % (noise_type, noise_param, test_performance_mean))
            File.flush()

    File.close()







