# -*- coding: utf-8 -*-
import os
import numpy as np
import argparse
import datetime
import re
import time
import torch
import torch.nn as nn
import random

import memory_lib


def get_arguments():
    parser = argparse.ArgumentParser(description='Description: run_RL')
    # RL
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.97)
    parser.add_argument('--entropy', type=float, default=0.10)
    parser.add_argument('--PPO_epochs', type=int, default=5)
    parser.add_argument('--eps_clip', type=float, default=0.2)
    parser.add_argument('--alg', type=str, default='ppo',
                        choices=['ppo', 'reinforce'])
    # Program
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--thread', type=int, default=-1)
    parser.add_argument('--train_num', type=int, default=20000)
    parser.add_argument('--rep', type=int, default=11)
    parser.add_argument('--ignore_checkpoint', default=False, action='store_true')
    parser.add_argument('--monitor_time', default=False, action='store_true')
    # Task amd model
    parser.add_argument('--task', type=str, default='gymip',
                        choices=['gymip'])
    parser.add_argument('--model', type=str, default='rwtaprob',
                        choices=['mlp3soft', 'mlp3relu', 'rwtaprob', 'rwtaspk', 'snnbptt', 'ann2snn'])
    parser.add_argument('--optimizer', type=str, default='rmsprop',
                        choices=['sgd', 'adam', 'rmsprop'])
    # ---------------------------------------------------
    # for Gym IP
    parser.add_argument('--gymip_train_xml', type=str, default='inverted_pendulum_ChangeThk_0.050000.xml')
    # for mlp3, snnbptt
    parser.add_argument('--hidden_num', type=int, default=64)
    # for rwta
    parser.add_argument('--hid_group_num', type=int, default=8)
    parser.add_argument('--hid_group_size', type=int, default=8)
    parser.add_argument('--rwta_del_connection', type=str, default='none',
                        choices=['none', 'hh', 'sa', 'hhsa', 'ha', 'sh'])
    # for rwtaspk
    parser.add_argument('--response_window', type=int, default=40)
    # for snnbptt
    parser.add_argument('--snn_num_steps', type=int, default=15)
    # ---------------------------------------------------
    return parser.parse_args()


def reload_log_file(filename):
    train_epi_num, val_best = None, None
    with open(filename) as file:
        for line in file:
            str_list = [i for i in re.sub(',', ' ', line).split()]
            if str_list[0] == 'train':
                train_epi_num = int(str_list[1])
            if str_list[0] == 'val':
                val_best = float(str_list[2])
    if train_epi_num is None:
        train_epi_num = 0
        val_best = -10000
    if val_best is None:
        val_best = -10000
    return train_epi_num, val_best


def log_text(file_handle, type_str, record_text, onscreen=True):
    global log_text_flush_time
    if onscreen:
        print('\033[92m%s\033[0m' % type_str.ljust(10), record_text)
    file_handle.write((type_str+',').ljust(10) + record_text + '\n')
    if time.time() - log_text_flush_time > 10:
        log_text_flush_time = time.time()
        file_handle.flush()
        os.fsync(file_handle.fileno())


class TimeMonitor:
    def __init__(self):
        self.size = 100
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
                        float(np.mean(self.time_inference)),
                        float(np.std(self.time_inference)),
                        np.min(self.time_inference),
                        np.max(self.time_inference), ))
        if rec_type == 2:
            self.time_optimize[self.time_pointer_optimize] = value * 1000
            self.time_pointer_optimize = (self.time_pointer_optimize + 1) % self.size
            if self.time_pointer_optimize == 0:
                print('timer opt: %7.3f %7.3f %7.3f %7.3f' % (
                        float(np.mean(self.time_optimize)),
                        float(np.std(self.time_optimize)),
                        np.min(self.time_optimize),
                        np.max(self.time_optimize), ))


if __name__ == "__main__":
    # Arguments
    args = get_arguments()
    if args.model in ['mlp3soft', 'mlp3relu']:
        model_str = 'h%d_-' % args.hidden_num
    elif args.model in ['snnbptt']:
        model_str = 'h%d_%d' % (args.hidden_num, args.snn_num_steps)
    elif args.model in ['rwtaprob']:
        model_str = 'h%d-%d_%s' % (args.hid_group_num, args.hid_group_size, args.rwta_del_connection)
    elif args.model in ['rwtaspk']:
        model_str = 'h%d-%d-%d_%s' % (args.hid_group_num, args.hid_group_size,
                                      args.response_window, args.rwta_del_connection)
    elif args.model in ['ann2snn']:
        model_str = 'h%d_-' % (args.hidden_num)
    else:
        model_str = 'error'
        print('\033[91mError in arguments\033[0m')
    # Pre-defined Parameters
    if args.task in ['gymip']:
        args.hidden_num, args.hid_group_num, args.hid_group_size = 64, 8, 8
        if args.alg == 'ppo':
            args.train_num = 2000
        else:
            args.train_num = 5000
    if args.task in ['mnist']:
        args.hidden_num, args.hid_group_num, args.hid_group_size = 200, 20, 10
        args.train_num = 10000
    if args.task in ['vizdoom']:
        args.hidden_num, args.hid_group_num, args.hid_group_size = 500, 50, 10
        if args.alg == 'ppo':
            args.train_num = 2000
        else:
            args.train_num = 5000
    EXP_NAME = '%s_%s_%s_%s_%s_%8.6f_%4.2f_%6.5f_%d_%5.4f_rep%02d' % (
            args.alg, args.task, args.model, model_str, args.optimizer,
            args.lr, args.entropy, args.gamma,
            args.PPO_epochs, args.eps_clip,
            args.rep)
    # Task specified variables
    if args.task in ['gymip',]:
        val_freq, val_num, test_num = 100, 10, 10
        train_frequency = 10
    elif args.task in ['vizdoom']:
        val_freq, val_num, test_num = 100, 10, 10
        train_frequency = 30
    elif args.task in ['mnist', 'cifar10']:
        val_freq, val_num, test_num = 100, 1, 1
        train_frequency = 5
    # Device
    if args.cuda < 0:
        torch_device = torch.device('cpu')
        if args.thread == -1:
            pass
        else:
            torch.set_num_threads(args.thread)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '%1d' % args.cuda
        torch_device = torch.device('cuda:0')
    # Environment Setup
    if args.task == 'mnist':
        import env_mnist
        env = env_mnist.MnistDataset(dev=torch_device)
        input_dimension, output_dimension = env.state_dim, env.action_num
        mem = memory_lib.MemoryBuffer(s_size=input_dimension, a_size=output_dimension, dev=torch_device)
    elif args.task == 'gymip':      # InvertedPendulum
        import env_gymip
        env = env_gymip.GymIP(train_xml_name=args.gymip_train_xml, dev=torch_device)
        input_dimension, output_dimension = env.state_dimension, env.action_num
        mem = memory_lib.MemoryBuffer(s_size=input_dimension, a_size=output_dimension, dev=torch_device)
    elif args.task == 'vizdoom':    # ViZDoom Health Gathering
        import env_vizdoom
        env = env_vizdoom.DoomHealthGathering(dev=torch_device)
        input_dimension, output_dimension = env.state_dimension, env.action_num
        mem = memory_lib.MemoryBuffer(s_size=input_dimension, a_size=output_dimension,
                                      memory_size=2000, batch_size=100, dev=torch_device)
    else:
        input_dimension, output_dimension = None, None
        env, mem = None, None
        print('Error in model name.')
    # Model Setup
    if args.model == 'mlp3soft':
        import model_mlp
        model = model_mlp.MLP_3(layer_sizes=[input_dimension, args.hidden_num, output_dimension],
                                hid_activate='softmax', hid_group_size=args.hid_group_size,
                                out_activate='softmax',
                                optimizer_name=args.optimizer, optimizer_learning_rate=args.lr,
                                entropy_ratio=args.entropy,)
    elif args.model == 'mlp3relu':
        import model_mlp
        model = model_mlp.MLP_3(layer_sizes=[input_dimension, args.hidden_num, output_dimension],
                                hid_activate='relu', hid_group_size=args.hid_group_size, 
                                out_activate='softmax',
                                optimizer_name=args.optimizer, optimizer_learning_rate=args.lr,
                                entropy_ratio=args.entropy, dev=torch_device)
    elif args.model == 'snnbptt':
        import model_snnbptt
        model = model_snnbptt.SNNBPTT3(
                layer_sizes=[input_dimension, args.hidden_num, output_dimension],
                snn_num_steps = args.snn_num_steps,
                optimizer_name=args.optimizer, optimizer_learning_rate=args.lr,
                entropy_ratio=args.entropy, dev=torch_device)
    elif args.model == 'rwtaprob':
        import model_rwta
        model = model_rwta.RWTAprob(input_size=input_dimension, output_size=output_dimension,
                                    hid_num=args.hid_group_num, hid_size=args.hid_group_size,
                                    remove_connection_pattern=args.rwta_del_connection,
                                    optimizer_name=args.optimizer, optimizer_learning_rate=args.lr,
                                    entropy_ratio=args.entropy, device=torch_device)
        mem.init_for_rwta(q_size=model.dim_has, v_size=model.dim_ha)
    elif args.model == 'rwtaspk':
        import model_rwta
        model = model_rwta.RWTAspike(
                input_size=input_dimension, output_size=output_dimension,
                hid_num=args.hid_group_num, hid_size=args.hid_group_size,
                spk_response_window='uni', spk_full_time=42, spk_resp_time=args.response_window,         # special for spiking version
                remove_connection_pattern=args.rwta_del_connection,
                optimizer_name=args.optimizer, optimizer_learning_rate=args.lr,
                entropy_ratio=args.entropy,
                device=torch_device)
        mem.init_for_rwta(q_size=model.dim_has, v_size=model.dim_ha)
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
        model_c = model_critic.Critic(input_size=input_dimension, output_size=output_dimension,
                                      dev=torch_device, small=True)
    else:
        model_c = model_critic.Critic(input_size=input_dimension, output_size=output_dimension,
                                      dev=torch_device)
    # Storage Folders
    if not os.path.exists('./log_model'):
        os.mkdir('./log_model')
    if not os.path.exists('./log_text'):
        os.mkdir('./log_text')
    model_current_save_time = time.time()
    log_text_flush_time = time.time()
    # Reload
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
    else:           # Initialize training
        last_train_epi_num, last_val_best = 0, -10000
        File = open(log_filename, 'w')
        log_text(File, 'init', str(datetime.datetime.now()))
        log_text(File, 'arguments', str(args))
        if args.model == 'ann2snn':
            model.load_model_ann(EXP_NAME + '_best')
    # Time Monitor
    calculation_time_monitor = TimeMonitor()
    # >>>>  Main Loop
    mem.reset()         # memory buffer is shared across episodes
    train_step_num_total = 0
    for train_epi_i in range((last_train_epi_num + 1), args.train_num):
        if args.model == 'ann2snn':
            break
        env.init_train()
        for train_step_i in range(env.max_step_num):
            observation = env.get_train_observation()
            # Inference
            if args.monitor_time:
                start_time = time.time()
            model_output, model_other_output = model(observation)
            if args.monitor_time:
                calculation_time_monitor.record_time(rec_type=1, value=(time.time()-start_time))
            # Process Output
            if args.model in ['rwtaprob', 'rwtaspk']:
                action_chosen_onehot = model_other_output[0]
                action_logprob = model_other_output[1]
            else:
                action_distribution = torch.distributions.OneHotCategorical(model_output)
                action_chosen_onehot = action_distribution.sample()
                action_logprob = action_distribution.log_prob(action_chosen_onehot)
            reward, observation_next, performance_list = env.make_action(action_chosen_onehot)
            if args.model in ['rwtaprob', 'rwtaspk']:
                mem.add_transition(s1=observation, model_output=model_output,
                                   a=action_chosen_onehot, a_log=action_logprob,
                                   r=reward, s2=observation_next, done=env.done_signal,
                                   q_has=model_other_output[2], v_ha=model_other_output[3])
            else:
                mem.add_transition(s1=observation, model_output=model_output.detach(),
                                   a=action_chosen_onehot, a_log=action_logprob.detach(),
                                   r=reward, s2=observation_next, done=env.done_signal)
            # >>>> Train
            train_step_num_total = (train_step_num_total + 1) % train_frequency     # every number of steps
            if train_step_num_total == 0:
                if args.model in ['rwtaprob', 'rwtaspk']:
                    s1, s2, model_output_1, a_1, a_logprob_1, r, done, q_has_1, v_ha_1 = mem.get_batch()
                else:
                    s1, s2, model_output_1, a_1, a_logprob_1, r, done = mem.get_batch()
                batch_size = s1.shape[0]
                model_output_2, model_other_output_2 = model(s2)
                s1_value = model_c(s1)
                s2_value = model_c(s2)
                a1_prob = model_output_1
                a2_prob = model_output_2
                s1_value_ave = torch.sum(a1_prob * s1_value, dim=1).detach()
                s2_value_ave = torch.sum(a2_prob * s2_value, dim=1).detach()
                # Update Critic
                state_value_target = s1_value.clone().detach()
                a1_index = torch.argmax(a_1, dim=1)     # onehot to index
                state_value_target[torch.arange(batch_size), a1_index] = \
                    r + (args.gamma * s2_value_ave) * (1 - done)
                model_c.learn(s1_value, state_value_target)
                # Update Agent
                advantage = (s1_value[torch.arange(batch_size), a1_index] - s1_value_ave).detach()
                if args.alg == 'ppo':
                    old_logprob = torch.clone(a_logprob_1)
                    for _ in range(args.PPO_epochs):
                        model_output_ppo, model_other_output_ppo = model(s1)
                        if args.model in ['rwtaprob', 'rwtaspk']:
                            model_output_ppo.requires_grad_()
                            action_distribution = torch.distributions.OneHotCategorical(model_output_ppo)
                            action_logprob_ppo = action_distribution.log_prob(a_1)
                        else:
                            action_distribution = torch.distributions.OneHotCategorical(model_output_ppo)
                            action_logprob_ppo = action_distribution.log_prob(a_1)
                        action_entropy = action_distribution.entropy()
                        # Optimization
                        if args.monitor_time:
                            start_time2 = time.time()
                        if args.model in ['rwtaprob', 'rwtaspk']:
                            model.learn_ppo(action_logprob_ppo, old_logprob, advantage,
                                    args.eps_clip, action_entropy,
                                    old_vha=v_ha_1, old_qhas=q_has_1, model_output=model_output_ppo,
                                    current_other=model_other_output_ppo,)
                        else:
                            model.learn_ppo(action_logprob_ppo, old_logprob, advantage,
                                    args.eps_clip, action_entropy,)
                        if args.monitor_time:
                            calculation_time_monitor.record_time(rec_type=2, value=(time.time()-start_time2))
                else:           # 'reinforce'
                    if args.model in ['rwtaprob', 'rwtaspk']:
                        model_output_rei = torch.clone(model_output_1)
                        model_output_rei.requires_grad_()
                        action_distribution = torch.distributions.OneHotCategorical(model_output_rei)
                        action_logprob_rei = action_distribution.log_prob(a_1)
                    else:
                        model_output_rei, model_other_output_rei = model(s1)
                        action_distribution = torch.distributions.OneHotCategorical(model_output_rei)
                        action_logprob_rei = action_distribution.log_prob(a_1)
                    action_entropy = action_distribution.entropy()
                    if args.monitor_time:
                        start_time2 = time.time()
                    if args.model in ['rwtaprob', 'rwtaspk']:
                        model.learn_reinforce(action_logprob_rei, advantage, action_entropy,
                                              v_ha=v_ha_1, q_has=q_has_1,
                                              model_output=model_output_rei)
                    else:
                        model.learn_reinforce(action_logprob_rei, advantage, action_entropy,)
                    if args.monitor_time:
                        calculation_time_monitor.record_time(rec_type=2, value=(time.time()-start_time2))
            # Episode End
            if env.done_signal == True:
                break
        # Checkpoint
        if time.time() - model_current_save_time > 10:
            model.save_model(EXP_NAME + '_current')
            model_c.save_model(EXP_NAME + 'critic' + '_current')
            model_current_save_time = time.time()
        log_text(File, 'train', '%d, %8.6f' % (train_epi_i, performance_list[0]), onscreen=False)
        # Validation
        if train_epi_i % val_freq == (val_freq - 1):
            val_preformance_list = []
            for val_epi_i in range(val_num):
                env.init_val()
                for val_step_i in range(env.max_step_num):
                    observation = env.get_val_observation()
                    model_output, model_other_output = model(observation)
                    action_chosen_index = torch.argmax(model_output, dim=1)
                    action_chosen_onehot = torch.nn.functional.one_hot(action_chosen_index, num_classes=env.action_num)
                    reward, observation_next, performance_list_val = env.make_action(action_chosen_onehot)
                    if env.done_signal == True:
                        break
                val_preformance_list.append(performance_list_val[0])
            val_performance_mean = sum(val_preformance_list) / len(val_preformance_list)
            if last_val_best <= val_performance_mean:
                model.save_model(EXP_NAME + '_best')
                model_c.save_model(EXP_NAME + 'critic' + '_best')
                log_text(File, 'val_save', '%d,   %8.6f' % (train_epi_i, val_performance_mean))
                last_val_best = val_performance_mean
            log_text(File, 'val', '%d,   %8.6f,   %8.6f' % (train_epi_i, val_performance_mean, performance_list[0]))


    # ANN2SNN Implementation
    if args.model == 'ann2snn':
        # Collect data
        print('ANN2SNN -> collect data for conversion')
        while True:
            if model.model_collect_full is True:
                break
            env.init_train()
            for train_step_i in range(env.max_step_num):
                observation = env.get_train_observation()
                model_output = model.ANN_model.get_prediction(observation)
                action_distribution = torch.distributions.OneHotCategorical(model_output)
                action_chosen_onehot = action_distribution.sample()
                action_logprob = action_distribution.log_prob(action_chosen_onehot)
                reward, observation_next, performance_list = env.make_action(action_chosen_onehot)
                model.add_s_list(observation)
                if env.done_signal is True:
                    break
        model.convert_model()



    # ~~~~~~~~~~~~~~~~~~~~TEST~~~~~~~~~~~~~~~~~~~~~~~~~~
    log_text(File, 'test', str(datetime.datetime.now()))

    # >>>> Test Adversarial
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
            observation = env.get_train_observation()
            model_output, model_other_output = model(observation)
            action_chosen_index = torch.argmax(model_output, dim=1)
            action_chosen_onehot = torch.nn.functional.one_hot(action_chosen_index, num_classes=env.action_num)
            reward, _, _ = env.make_action(action_chosen_onehot)
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
        log_text(File, 'ADtrain', '%8d,   %8.6f' % (collect_epi_i, ad_loss), onscreen=False)
    perturb_loss = nn.CrossEntropyLoss()
    model.load_model(EXP_NAME + '_best')
    for epsilon in np.arange(0, 0.2, 0.01):
        test_preformance_list = []
        for test_epi_i in range(test_num):
            env.init_test()
            for test_step_i in range(env.max_step_num):
                observation = env.get_test_observation()
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
                reward, _, test_other_step_record = env.make_action(action_chosen_onehot)
                if env.done_signal == True:
                    break
            test_preformance_list.append(test_other_step_record[0])
        test_performance_mean = sum(test_preformance_list) / len(test_preformance_list)
        log_text(File, 'FGSM', '%8.6f,   %8.6f' % (epsilon, test_performance_mean))
        File.flush()
    
    
    # >>>> Test Weight ABS
    noise_type_list = ['gaussian', 'uniform']
    for noise_type in noise_type_list:
        # Noise Parameters
        if noise_type in ['gaussian']:
            noise_param_list = np.arange(0, 1.0, 0.02)
        else:    # if noise_type in ['uniform']:
            noise_param_list = np.arange(0, 4.0, 0.05)
        for noise_param in noise_param_list:
            test_preformance_list = []
            for test_epi_i in range(test_num):
                model.load_model(EXP_NAME + '_best')
                model.add_noise_abs(noise_type, noise_param)
                env.init_test()
                for test_step_i in range(env.max_step_num):
                    observation = env.get_test_observation()
                    model_output, model_other_output = model(observation)
                    action_chosen_index = torch.argmax(model_output, dim=1)
                    action_chosen_onehot = torch.nn.functional.one_hot(action_chosen_index, num_classes=env.action_num)
                    reward, _, test_other_step_record = env.make_action(action_chosen_onehot)
                    if env.done_signal == True:
                        break
                test_preformance_list.append(test_other_step_record[0])
            test_performance_mean = sum(test_preformance_list) / len(test_preformance_list)
            log_text(File, 'w_noise', '%s,  %8.6f,   %8.6f' % (noise_type, noise_param, test_performance_mean))
            File.flush()


    # >>>> Test Weight REL
    noise_type_list = ['gaussian', 'uniform']
    for noise_type in noise_type_list:
        # Noise Parameters
        if noise_type in ['gaussian']:
            noise_param_list = np.arange(0, 5, 0.1)
        else:    # if noise_type in ['uniform']:
            noise_param_list = np.arange(0, 5, 0.1)
        for noise_param in noise_param_list:
            test_preformance_list = []
            for test_epi_i in range(test_num):
                model.load_model(EXP_NAME + '_best')
                model.add_noise_relative(noise_type, noise_param)
                env.init_test()
                for test_step_i in range(env.max_step_num):
                    observation = env.get_test_observation()
                    model_output, model_other_output = model(observation)
                    action_chosen_index = torch.argmax(model_output, dim=1)
                    action_chosen_onehot = torch.nn.functional.one_hot(action_chosen_index, num_classes=env.action_num)
                    reward, _, test_other_step_record = env.make_action(action_chosen_onehot)
                    if env.done_signal == True:
                        break
                test_preformance_list.append(test_other_step_record[0])
            test_performance_mean = sum(test_preformance_list) / len(test_preformance_list)
            log_text(File, 'w_rel_noise', '%s,  %8.6f,   %8.6f' % (noise_type, noise_param, test_performance_mean))
            File.flush()

    # >>>> Test Input
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
            for test_epi_i in range(test_num):
                env.init_test()
                for test_step_i in range(env.max_step_num):
                    observation = env.get_test_observation(noise_type=noise_type, noise_param=noise_param)
                    model_output, model_other_output = model(observation)
                    action_chosen_index = torch.argmax(model_output, dim=1)
                    action_chosen_onehot = torch.nn.functional.one_hot(action_chosen_index, num_classes=env.action_num)
                    reward, _, test_other_step_record = env.make_action(action_chosen_onehot)

                    if env.done_signal == True:
                        break
                test_preformance_list.append(test_other_step_record[0])
            test_performance_mean = sum(test_preformance_list) / len(test_preformance_list)
            log_text(File, 'i_noise', '%s,  %8.6f,   %8.6f' % (noise_type, noise_param, test_performance_mean))
            File.flush()

    # >>>> Test GYMIP / GYMDIP Env
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
                for test_epi_i in range(test_num):
                    env.init_test(variation_type=noise_type, variation_param=noise_param)
                    for test_step_i in range(env.max_step_num):
                        observation = env.get_test_observation()
                        model_output, model_other_output = model(observation)
                        action_chosen_index = torch.argmax(model_output, dim=1)
                        action_chosen_onehot = torch.nn.functional.one_hot(action_chosen_index, num_classes=env.action_num)
                        reward, _, test_other_step_record = env.make_action(action_chosen_onehot)

                        if env.done_signal == True:
                            break
                    test_preformance_list.append(test_other_step_record[0])
                test_performance_mean = sum(test_preformance_list) / len(test_preformance_list)
                log_text(File, 'e_gymip', '%s,  %8.6f,   %8.6f' % (noise_type, noise_param, test_performance_mean))
                File.flush()

    # >>>> Test RWTA Connection Knock Out
    if args.model in ['rwtaprob', 'rwtaspk']:
        noise_type_list = ['hh', 'sh', 'sa', 'ha', 'all']
        noise_param_list = np.arange(0, 0.5, 0.02)
        for noise_type in noise_type_list:
            for noise_param in noise_param_list:
                test_preformance_list = []
                for test_epi_i in range(test_num):
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
                        observation = env.get_test_observation()
                        model_output, model_other_output = model(observation)
                        action_chosen_index = torch.argmax(model_output, dim=1)
                        action_chosen_onehot = torch.nn.functional.one_hot(action_chosen_index, num_classes=env.action_num)
                        reward, _, test_other_step_record = env.make_action(action_chosen_onehot)
                        if env.done_signal == True:
                            break
                    test_preformance_list.append(test_other_step_record[0])
                test_performance_mean = sum(test_preformance_list) / len(test_preformance_list)
                log_text(File, 'RWTA_conn', '%s,  %8.6f,   %8.6f' % (noise_type, noise_param, test_performance_mean))
                File.flush()

    File.close()







