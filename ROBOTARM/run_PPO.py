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
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.97)
    parser.add_argument('--entropy', type=float, default=0.05)
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--thread', type=int, default=-1)
    parser.add_argument('--PPO_epochs', type=int, default=5)
    parser.add_argument('--eps_clip', type=float, default=0.2)

    parser.add_argument('--task', type=str, default='robotarm',
            choices=['mnist', 'cifar10', 'vizdoom', 'gymip', 'gymdip', 'robotarm'])
    parser.add_argument('--model', type=str, default='mlp3relu',
            choices=['mlp3soft', 'mlp3relu', 'rwtaprob', 'rwtaspk', 'snnbptt', 'ann2snn'])
    parser.add_argument('--optimizer', type=str, default='adam',
            choices=['sgd', 'adam', 'rmsprop'])
    # ---------------------------------------------------
    # for GYMIP
    parser.add_argument('--gymip_train_xml', type=str, default='inverted_pendulum_ChangeThk_0.050000.xml')
    # for mlp3, snnbptt
    parser.add_argument('--hidden_num', type=int, default=1000)
    # for rwta
    parser.add_argument('--hid_group_num', type=int, default=100)
    parser.add_argument('--hid_group_size', type=int, default=10)
    parser.add_argument('--rwta_del_connection', type=str, default='none',
            choices=['none', 'hh', 'sa', 'hhsa', 'ha', 'sh'])
    # for rwtaspk
    parser.add_argument('--response_window', type=int, default=20)
    # for snnbptt
    parser.add_argument('--snn_num_steps', type=int, default=25)
    # ---------------------------------------------------
    parser.add_argument('--train_num', type=int, default=20000)
    parser.add_argument('--val_freq', type=int, default=100)
    parser.add_argument('--val_num', type=int, default=10)
    parser.add_argument('--test_num', type=int, default=10)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--ignore_checkpoint', default=False, action='store_true')
    parser.add_argument('--monitor_time', default=False, action='store_true')
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
    if train_epi_num is None:
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


class time_monitor:
    def __init__(self):
        self.size = 500
        self.time_pointer_inference = 0
        self.time_pointer_optimize = 0
        self.time_inference = np.ones([self.size], dtype=np.float32) * (-1)
        self.time_optimize = np.ones([self.size], dtype=np.float32) * (-1)

    def record_time(self, rec_type=1, value=0.0):
        if rec_type == 1:
            self.time_inference[self.time_pointer_inference] = value * 1000
            self.time_pointer_inference = (self.time_pointer_inference + 1) % self.size
            if self.time_pointer_inference == 0:
                print('timer inf: %7.3f %7.3f %7.3f %7.3f' % (
                        np.mean(self.time_inference),
                        np.std(self.time_inference),
                        np.min(self.time_inference),
                        np.max(self.time_inference), ))
        if rec_type == 2:
            self.time_optimize[self.time_pointer_optimize] = value * 1000
            self.time_pointer_optimize = (self.time_pointer_optimize + 1) % self.size
            if self.time_pointer_optimize == 0:
                print('timer opt: %7.3f %7.3f %7.3f %7.3f' % (
                        np.mean(self.time_optimize),
                        np.std(self.time_optimize),
                        np.min(self.time_optimize),
                        np.max(self.time_optimize), ))


if __name__ == "__main__":
    # -----arguments---------
    args = get_arguments()
    if args.model in ['mlp3soft', 'mlp3relu',]:
        model_hyperparam_str = 'h%d' % (args.hidden_num)
    elif args.model in ['snnbptt']:
        model_hyperparam_str = 'h%d_%d' % (args.hidden_num, args.snn_num_steps)
    elif args.model in ['rwtaprob']:
        model_hyperparam_str = 'h%d-%d_%s' % (args.hid_group_num, args.hid_group_size, args.rwta_del_connection)
    elif args.model in ['rwtaspk']:
        model_hyperparam_str = 'h%d-%d-%d_%s' % (args.hid_group_num, args.hid_group_size,
                                                 args.response_window, args.rwta_del_connection)
    elif args.model in ['ann2snn']:
        model_hyperparam_str = 'h%d' % (args.hidden_num,)
    else:
        print('\033[91mError in arguments\033[0m')
    EXP_NAME = 'PPO_%s_%s_%s_%s_%8.6f_%4.2f_%6.5f_%d_%5.4f_rep%02d' % (
            args.task, args.model,
            model_hyperparam_str,
            args.optimizer, args.lr, args.entropy, args.gamma,
            args.PPO_epochs, args.eps_clip,
            args.seed)
    # -----cuda device-------
    if args.thread == -1:
        pass
    else:
        torch.set_num_threads(args.thread)
    if args.cuda < 0:
        torch_device = torch.device('cpu')
    else:
        torch_device = torch.device('cuda:%d' % (args.cuda))
    # -----env---------------
    if args.task == 'robotarm':
        import env_robotarm
        env = env_robotarm.ROBOTARM_CONTROL()
        if args.cuda < 0:
            env.convert_all_to_torch_cpu()
        else:
            env.convert_all_to_torch_gpu(args.cuda)
        input_dimension, output_dimension = env.state_dimension, env.action_num
        train_batch_size, test_batch_size = 1, 1
        mem = memory_lib.MemoryList(gamma=args.gamma)
    # -----model-------------
    if args.model == 'mlp3soft':
        import model_mlp
        model = model_mlp.MLP_3(
                layer_sizes=[input_dimension, args.hidden_num, output_dimension],
                hid_activate='softmax', hid_group_size=args.hid_group_size,
                out_activate='softmax', optimizer_name=args.optimizer, optimizer_learning_rate=args.lr,
                entropy_ratio=args.entropy,)
    elif args.model == 'mlp3relu':
        import model_mlp
        model = model_mlp.MLP_3(
                layer_sizes=[input_dimension, args.hidden_num, output_dimension],
                hid_activate='relu', hid_group_size=args.hid_group_size,
                out_activate='softmax', optimizer_name=args.optimizer, optimizer_learning_rate=args.lr,
                entropy_ratio=args.entropy,)
    elif args.model == 'snnbptt':
        import model_snnbptt
        model = model_snnbptt.SNNBPTT3(
                layer_sizes=[input_dimension, args.hidden_num, output_dimension],
                snn_num_steps = args.snn_num_steps,
                optimizer_name=args.optimizer, optimizer_learning_rate=args.lr,
                entropy_ratio=args.entropy,)
    elif args.model == 'rwtaprob':
        import model_rwta
        model = model_rwta.RWTA_Prob(
                input_size=input_dimension, output_size=output_dimension,
                hid_num=args.hid_group_num, hid_size=args.hid_group_size,
                remove_connection_pattern=args.rwta_del_connection,
                optimizer_name=args.optimizer, optimizer_learning_rate=args.lr,
                entropy_ratio=args.entropy,
                device=torch_device)
    elif args.model == 'rwtaspk':
        import model_rwta
        model = model_rwta.RWTA_Spike(
                input_size=input_dimension, output_size=output_dimension,
                hid_num=args.hid_group_num, hid_size=args.hid_group_size,
                spk_response_window='uni', spk_full_time=200, spk_resp_time=args.response_window,         # special for spiking version
                remove_connection_pattern=args.rwta_del_connection,
                optimizer_name=args.optimizer, optimizer_learning_rate=args.lr,
                entropy_ratio=args.entropy,
                device=torch_device)
    elif args.model == 'ann2snn':
        import model_convert
        model = model_convert.MLP_3(
                layer_sizes=[input_dimension, args.hidden_num, output_dimension],
                hid_activate='relu', hid_group_size=args.hid_group_size,
                out_activate='softmax', optimizer_name=args.optimizer, optimizer_learning_rate=args.lr,
                snn_num_steps = args.snn_num_steps,
                entropy_ratio=args.entropy, device=torch_device)
    import model_critic
    if args.task in ['gymip', 'gymdip']:
        model_c = model_critic.Critic(input_size=input_dimension, output_size=1, small=True)
    else:
        model_c = model_critic.Critic(input_size=input_dimension, output_size=1)

    # -----init timer--------
    model_current_save_time = time.time()
    log_text_flush_time = time.time()
    # -----reload------------
    reload_data = True
    log_filename = './log_text/log_' + EXP_NAME + '.txt'
    if not os.path.exists(log_filename):
        reload_data = False
    if not os.path.exists('./log_model/' + EXP_NAME + '_current_1.pt'):
        if not os.path.exists('./log_model/' + EXP_NAME + '_current_b_1.pt'):
            reload_data = False
    if args.ignore_checkpoint == True:
        reload_data = False
    if args.model == 'ann2snn':
        reload_data = False
    if reload_data:
        last_train_epi_num, last_val_best = reload_log_file(log_filename)
        File = open(log_filename, 'a')
        log_text(File, 'resume', str(datetime.datetime.now()))
        model.load_model(EXP_NAME + '_current')
        model_c.load_model(EXP_NAME + 'critic' + '_current')
    else:
        # -----initialize--------
        last_train_epi_num, last_val_best = 0, -10000
        File = open(log_filename, 'w')
        log_text(File, 'init', str(datetime.datetime.now()))
        log_text(File, 'arguments', str(args))
        if args.model == 'ann2snn':
            model.load_model_ann(EXP_NAME + '_best')
    # -----monitor time------
    calculation_time_monitor = time_monitor()
    # -----main loop---------
    model = model.to(torch_device)
    model_c = model_c.to(torch_device)
    for train_epi_i in range((last_train_epi_num + 1), args.train_num):
        if args.model == 'ann2snn':
            break
        env.init_train()
        mem.reset()
        for train_step_i in range(env.max_step_num):
            observation = env.get_train_observation(batch_size=train_batch_size)
            if args.monitor_time:
                start_time = time.time()
            model_output, model_other_output = model(observation)
            if args.monitor_time:
                calculation_time_monitor.record_time(rec_type=1, value=(time.time()-start_time))
            if args.model in ['rwtaprob', 'rwtaspk']:
                action_chosen_onehot = model_other_output[0]
                action_logprob = model_other_output[1]
            else:
                action_distribution = torch.distributions.OneHotCategorical(model_output)
                action_chosen_onehot = action_distribution.sample()
                action_logprob = action_distribution.log_prob(action_chosen_onehot)
            
            # KEYBOARD CHECK
            ''' CHECK -- KEYBOARD CONTROL -- CUDA
            print('Input: ')
            action_chosen_onehot = torch.zeros([1, 5]).cuda()
            while True:
                com = input()
                if com == 'w':
                    action_chosen_onehot[0, 0] = 1; break
                elif com == 's':
                    action_chosen_onehot[0, 1] = 1; break
                elif com == 'a':
                    action_chosen_onehot[0, 2] = 1; break
                elif com == 'd':
                    action_chosen_onehot[0, 3] = 1; break
                elif com == 'f':
                    action_chosen_onehot[0, 4] = 1; break
                # elif com == 'r':
                #     action_chosen_onehot[0, 5] = 1; break
                elif com == '1':
                    quit()
            # print(model_output)
            # raise Exception('Check')
            '''
            
            
            reward, train_other_step_record = env.make_action(action_chosen_onehot)
            mem.add_transition(s1=observation,
                    a_output=model_output,
                    a_logprob=action_logprob,
                    a_onehot=action_chosen_onehot,
                    r=reward, s2=None, other=model_other_output)
            if env.done_signal == True:
                break
        # -----Train-------------
        if args.task in ['gymip', 'gymdip']: # 'robotarm'
            mem.tune_reward(reward_normalization=False)
        else:
            mem.tune_reward(reward_normalization=True)          # Episode-level normalization
        s1, a_output, a_logprob, a_onehot, q, s2, old_other = mem.get_batch()
        old_logprob = torch.clone(a_logprob)
        for _ in range(args.PPO_epochs):
            # Evaluate ----------
            state_value = model_c(s1)
            model_output, model_other_output = model(s1)
            if args.model in ['rwtaprob', 'rwtaspk']:
                model_output.requires_grad_()
                action_distribution = torch.distributions.OneHotCategorical(model_output)
                action_logprob = action_distribution.log_prob(a_onehot)
            else:
                action_distribution = torch.distributions.OneHotCategorical(model_output)
                action_logprob = action_distribution.log_prob(a_onehot)
            action_entropy = action_distribution.entropy()
            q_add_dim = torch.unsqueeze(q, dim=1)
            advantage = q_add_dim - state_value.detach()
            loss_value_c = model_c.learn(state_value, q_add_dim)
            if args.monitor_time:
                start_time2 = time.time()
            loss_value = model.learn_episode_ppo(action_logprob, old_logprob, advantage,
                    args.eps_clip, action_entropy,
                    other=[old_other, model_other_output], model_output=model_output, a_onehot=a_onehot)
            if args.monitor_time:
                calculation_time_monitor.record_time(rec_type=2, value=(time.time()-start_time2))
            loss_value = loss_value.detach().item()
        if time.time() - model_current_save_time > 10:
            model.save_model(EXP_NAME + '_current')
            model_c.save_model(EXP_NAME + 'critic' + '_current')
            model_current_save_time = time.time()
        # log_text(File, 'train', '%d, %8.6f, %8.6f' % (train_epi_i, train_other_step_record[0], loss_value), onscreen=False)
        log_text(File, 'train', '%d, %8.6f, %8.6f' % (train_epi_i, train_other_step_record[0], loss_value), onscreen=True)
        # -----validate----------
        if train_epi_i % args.val_freq == (args.val_freq - 1):
            val_preformance_list = []
            for val_epi_i in range(args.val_num):
                env.init_val()
                for val_step_i in range(env.max_step_num):
                    observation = env.get_val_observation(batch_size=train_batch_size)
                    model_output, model_other_output = model(observation)
                    action_chosen_index = torch.argmax(model_output, dim=1)
                    action_chosen_onehot = torch.nn.functional.one_hot(action_chosen_index, num_classes=env.action_num)
                    reward, val_other_step_record = env.make_action(action_chosen_onehot)
                    if env.done_signal == True:
                        break
                """ Note other_step_record:
                        MNIST -- accuracy
                """
                val_preformance_list.append(val_other_step_record[0])
            val_performance_mean = sum(val_preformance_list) / len(val_preformance_list)
            if last_val_best <= val_performance_mean:           # 20230213 "<" --> "<="
                model.save_model(EXP_NAME + '_best')
                model_c.save_model(EXP_NAME + 'critic' + '_best')
                log_text(File, 'val_save', '%d,   %8.6f' % (train_epi_i, val_performance_mean))
                last_val_best = val_performance_mean
            log_text(File, 'val', '%d,   %8.6f,   %8.6f' % (train_epi_i, val_performance_mean, train_other_step_record[0]))


    # ~~~~~~~~~~~~~~~~~~~~ANN2SNN~~~~~~~~~~~~~~~~~~~~~~~
    if args.model == 'ann2snn':
        # Collect data
        print('ANN2SNN -> collect data for conversion')
        while True:
            if model.model_collect_full is True:
                break
            env.init_train()
            for train_step_i in range(env.max_step_num):
                observation = env.get_train_observation(batch_size=train_batch_size)
                model_output = model.ANN_model.get_prediction(observation)
                action_distribution = torch.distributions.OneHotCategorical(model_output)
                action_chosen_onehot = action_distribution.sample()
                action_logprob = action_distribution.log_prob(action_chosen_onehot)
                reward, train_other_step_record = env.make_action(action_chosen_onehot)
                model.add_s_list(observation)
                if env.done_signal is True:
                    break
        model.convert_model()



    # ~~~~~~~~~~~~~~~~~~~~TEST~~~~~~~~~~~~~~~~~~~~~~~~~~
    log_text(File, 'test', str(datetime.datetime.now()))


    '''
    # ~~~~~~~~~~~~~~~~~~~~TEST ADVERSARIAL~~~~~~~~~~~~~~
    import model_adversarial
    ad_mem_size = 1000
    if args.task in ['mnist', 'cifar10']:
        ad_train_epi_num = 1000
    else:               # 'vizdoom', 'gymip'
        ad_train_epi_num = 100
    ad_mem_s = torch.zeros([ad_mem_size, input_dimension]).to(torch_device)
    ad_mem_a = torch.zeros([ad_mem_size, output_dimension]).to(torch_device)
    ad_model = model_adversarial.Adversarial(input_dimension, output_dimension)
    ad_model = ad_model.to(torch_device)
    pointer, total_num = 0, 0
    model.load_model(EXP_NAME + '_best')
    for collect_epi_i in range(ad_train_epi_num):
        env.init_train()
        for collect_step_i in range(env.max_step_num):
            observation = env.get_train_observation(batch_size=train_batch_size)
            model_output, model_other_output = model(observation)
            action_chosen_index = torch.argmax(model_output, dim=1)
            action_chosen_onehot = torch.nn.functional.one_hot(action_chosen_index, num_classes=env.action_num)
            reward, _ = env.make_action(action_chosen_onehot)
            for sample_i in range(observation.shape[0]):
                ad_mem_s[pointer, :] = observation[sample_i, :]
                ad_mem_a[pointer, :] = action_chosen_onehot[sample_i, :]
                pointer = (pointer + 1) % ad_mem_size
                total_num = min(ad_mem_size, total_num + 1)
            if env.done_signal == True:
                break
        sample_list = random.sample(range(0, total_num), min(total_num, 200))
        ad_s_batch = ad_mem_s[sample_list]
        ad_a_batch = ad_mem_a[sample_list]
        ad_predict = ad_model(ad_s_batch)
        ad_loss = ad_model.learn(ad_s_batch, ad_predict, ad_a_batch)
        log_text(File, 'ADtrain', '%8d,   %8.6f' % (collect_epi_i, ad_loss))
    perturb_loss = nn.CrossEntropyLoss()
    model.load_model(EXP_NAME + '_best')
    for epsilon in np.arange(0, 0.2, 0.01):
        test_preformance_list = []
        for test_epi_i in range(args.test_num):
            env.init_test()
            for test_step_i in range(env.max_step_num):
                observation = env.get_test_observation(batch_size=test_batch_size)
                observation.requires_grad = True            # for FGSM
                ad_output = ad_model(observation)
                ad_argmax = torch.argmax(ad_output, dim=1)
                perturb_loss_value = perturb_loss(ad_output, ad_argmax).mean()
                ad_model.zero_grad()
                perturb_loss_value.backward()
                observation_grad = observation.grad.data
                sign_grad = observation_grad.sign()
                observation_perturb = observation + epsilon * sign_grad

                model_output, model_other_output = model(observation_perturb)
                action_chosen_index = torch.argmax(model_output, dim=1)
                action_chosen_onehot = torch.nn.functional.one_hot(action_chosen_index, num_classes=env.action_num)
                reward, test_other_step_record = env.make_action(action_chosen_onehot)
                if env.done_signal == True:
                    break
            test_preformance_list.append(test_other_step_record[0])
        test_performance_mean = sum(test_preformance_list) / len(test_preformance_list)
        log_text(File, 'FGSM', '%8.6f,   %8.6f' % (epsilon, test_performance_mean))
        File.flush()
    '''


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
                model.load_model(EXP_NAME + '_best')
                model.add_noise_abs(noise_type, noise_param)
                env.init_test()
                for test_step_i in range(env.max_step_num):
                    observation = env.get_test_observation(batch_size=test_batch_size)
                    model_output, model_other_output = model(observation)
                    action_chosen_index = torch.argmax(model_output, dim=1)
                    action_chosen_onehot = torch.nn.functional.one_hot(action_chosen_index, num_classes=env.action_num)
                    reward, test_other_step_record = env.make_action(action_chosen_onehot)
                    if env.done_signal == True:
                        break
                test_preformance_list.append(test_other_step_record[0])
            test_performance_mean = sum(test_preformance_list) / len(test_preformance_list)
            log_text(File, 'w_noise', '%s,  %8.6f,   %8.6f' % (noise_type, noise_param, test_performance_mean))
            File.flush()


    # -----test weight rel---       RELATIVE WEIGHT PERTURBATION
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
                model.load_model(EXP_NAME + '_best')
                model.add_noise_relative(noise_type, noise_param)
                env.init_test()
                for test_step_i in range(env.max_step_num):
                    observation = env.get_test_observation(batch_size=test_batch_size)
                    model_output, model_other_output = model(observation)
                    action_chosen_index = torch.argmax(model_output, dim=1)
                    action_chosen_onehot = torch.nn.functional.one_hot(action_chosen_index, num_classes=env.action_num)
                    reward, test_other_step_record = env.make_action(action_chosen_onehot)
                    if env.done_signal == True:
                        break
                test_preformance_list.append(test_other_step_record[0])
            test_performance_mean = sum(test_preformance_list) / len(test_preformance_list)
            log_text(File, 'w_rel_noise', '%s,  %8.6f,   %8.6f' % (noise_type, noise_param, test_performance_mean))
            File.flush()

    # -----test input--------
    model.load_model(EXP_NAME + '_best')
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
                    model_output, model_other_output = model(observation)
                    action_chosen_index = torch.argmax(model_output, dim=1)
                    action_chosen_onehot = torch.nn.functional.one_hot(action_chosen_index, num_classes=env.action_num)
                    reward, test_other_step_record = env.make_action(action_chosen_onehot)

                    if env.done_signal == True:
                        break
                test_preformance_list.append(test_other_step_record[0])
            test_performance_mean = sum(test_preformance_list) / len(test_preformance_list)
            log_text(File, 'i_noise', '%s,  %8.6f,   %8.6f' % (noise_type, noise_param, test_performance_mean))
            File.flush()

    '''
    # -----test gymip env----       # added gymdip
    if args.task in ['gymip', 'gymdip']:
        model.load_model(EXP_NAME + '_best')
        noise_type_list = ['length', 'thick', 'union']
        if args.task == 'gymdip':
            noise_type_list = ['thick']
        for noise_type in noise_type_list:
            if noise_type == 'length':
                noise_param_list = np.arange(0.16, 4.88, 0.08)
            elif noise_type == 'thick':
                noise_param_list = np.arange(0.01, 0.305, 0.005)
            else:
                noise_param_list = np.arange(0.02, 0.305, 0.005)
            for noise_param in noise_param_list:
                test_preformance_list = []
                for test_epi_i in range(args.test_num):
                    env.init_test(variation_type=noise_type, variation_param=noise_param)
                    for test_step_i in range(env.max_step_num):
                        observation = env.get_test_observation()
                        model_output, model_other_output = model(observation)
                        action_chosen_index = torch.argmax(model_output, dim=1)
                        action_chosen_onehot = torch.nn.functional.one_hot(action_chosen_index, num_classes=env.action_num)
                        reward, test_other_step_record = env.make_action(action_chosen_onehot)

                        if env.done_signal == True:
                            break
                    test_preformance_list.append(test_other_step_record[0])
                test_performance_mean = sum(test_preformance_list) / len(test_preformance_list)
                log_text(File, 'e_gymip', '%s,  %8.6f,   %8.6f' % (noise_type, noise_param, test_performance_mean))
                File.flush()
    '''

    '''
    # -----test RWTA conn----       # randomly remove some weights in RWTA
    if args.model in ['rwtaprob', 'rwtaspk']:
        noise_type_list = ['hh', 'sh', 'sa', 'ha', 'all']
        noise_param_list = np.arange(0, 0.5, 0.01)
        for noise_type in noise_type_list:
            for noise_param in noise_param_list:
                test_preformance_list = []
                for test_epi_i in range(args.test_num):
                    model.load_model(EXP_NAME + '_best')
                    if noise_type == 'all':
                        model.random_remove_weight('sh', noise_param)
                        model.random_remove_weight('sa', noise_param)
                        model.random_remove_weight('hh', noise_param)
                        model.random_remove_weight('ha', noise_param)
                    else:
                        model.random_remove_weight(noise_type, noise_param)
                    env.init_test()
                    for test_step_i in range(env.max_step_num):
                        observation = env.get_test_observation(batch_size=test_batch_size)
                        model_output, model_other_output = model(observation)
                        action_chosen_index = torch.argmax(model_output, dim=1)
                        action_chosen_onehot = torch.nn.functional.one_hot(action_chosen_index, num_classes=env.action_num)
                        reward, test_other_step_record = env.make_action(action_chosen_onehot)
                        if env.done_signal == True:
                            break
                    test_preformance_list.append(test_other_step_record[0])
                test_performance_mean = sum(test_preformance_list) / len(test_preformance_list)
                log_text(File, 'RWTA_conn', '%s,  %8.6f,   %8.6f' % (noise_type, noise_param, test_performance_mean))
                File.flush()
    '''

    File.close()







