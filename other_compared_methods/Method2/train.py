

# BASED ON https://github.com/asneha213/spiking-agent-RL/tree/master
# CHANGED TO TRAIN ON GYMIP AND TEST WITH ENVIRONMENT VARIATIONS


import numpy as np
import math
import argparse
import random
import itertools
import pdb
from matplotlib import pyplot as plt
import pickle

import time
import datetime
import os
import argparse

import env_gymip



#Constants

mod_F = 10
m_c = 1
m_p = 0.1
l = 0.5
td = 0.02
g = 9.8


def log_text(file_handle, type_str, record_text, onscreen=True):
    global log_text_flush_time
    if onscreen:
        print('\033[92m%s\033[0m' % type_str.ljust(10), record_text)
    file_handle.write((type_str+',').ljust(10) + record_text + '\n')
    if time.time() - log_text_flush_time > 10:
        log_text_flush_time = time.time()
        file_handle.flush()
        os.fsync(file_handle.fileno())


class ActorCritic():
    def __init__(self, order, epsilon, step_size, sigma=0.1, num_states=4, radial_sigma=None):
        self.num_states = num_states
        self.epsilon = epsilon
        self.alpha = step_size
        self.sigma = sigma
        # self.cartpole = CartPole()
        self.env = env_gymip.GymIP(train_xml_name='inverted_pendulum_ChangeThk_0.050000.xml')
        
        self.order = order
        self.lda = 0.5
        self.w = {}

        self.w[-1] = 5*np.ones(int(math.pow(order+1, num_states)))
        self.w[1] = 5*np.ones(int(math.pow(order+1, num_states)))
        
        self.combns = np.array(list(itertools.product(range(order+1), repeat=num_states)))
        '''
        self.x_lim = [-3,3]
        self.v_lim = [-10,10]
        self.theta_lim = [-math.pi/2,math.pi/2]
        self.omega_lim = [-math.pi, math.pi]
        '''
        # GYMIP, SATE ALREADY CLIPPED TO [-1, 1]
        self.x_lim = [-1, 1]
        self.v_lim = [-1, 1]
        self.theta_lim = [-1, 1]
        self.omega_lim = [-1, 1]
        self.actors = [SpikingActor() for i in range(10)]


    def fourier_feature_state(self, state, method='fourier'):
        state_norm = np.zeros(self.num_states)
        # CLIP (ALREADY CLIPPED TO [0, 1])
        state_norm[0] = (state[0]+self.x_lim[1])/(self.x_lim[1]-self.x_lim[0])
        state_norm[1] = (state[1]+self.v_lim[1])/(self.v_lim[1]-self.v_lim[0])
        state_norm[2] = (state[2]+self.theta_lim[1])/(self.theta_lim[1]-self.theta_lim[0])
        state_norm[3] = (state[3]+self.omega_lim[1])/(self.omega_lim[1]-self.omega_lim[0])

        prod_array = np.array([np.dot(state_norm, i) for i in self.combns])
        features = np.array(np.cos(np.pi*prod_array))
        return features


    # def e_greedy_action(self, action_ind):
    #     prob = (self.epsilon/2)*np.ones(2)
    #     prob[action_ind] = (1 - self.epsilon) + (self.epsilon/2)
    #     #e_action = 2*np.random.choice(2,1,p=prob)-1
    #     pr_array = np.concatenate((np.ones(int(100*prob[1])), -1*np.ones(int(100*prob[0]))))
    #     e_action = pr_array[random.randint(0, len(pr_array)-1)]
    #     return int(e_action)


    # def softmax_selection(self, qvalues, sigma):
    #     eps = 1e-5
    #     qvalues = qvalues + eps
    #     prob = np.exp(sigma*qvalues)/sum(np.exp(sigma*qvalues))
    #     prob[1] = 1-prob[0]
    #     e_action = 2*np.random.choice(2,1,p=prob)-1
    #     return int(e_action)

    def save_model(self, name):
        weight_list = []
        for actor_i in range(len(self.actors)):
            temp_actor = self.actors[actor_i]
            weight_list.append([temp_actor.ih_weights, temp_actor.ho_weights])
        with open('./log_model/model_' + name + '.pkl', 'wb') as temp_file:
            pickle.dump(weight_list, temp_file)
    
    def load_model(self, name):
        with open('./log_model/model_' + name + '.pkl', 'rb') as temp_file:
            weight_list = pickle.load(temp_file)
        for actor_i in range(len(self.actors)):
            self.actors[actor_i].ih_weights = weight_list[actor_i][0]
            self.actors[actor_i].ho_weights = weight_list[actor_i][1]

    def get_action_from_state(self, state_in, temp_count):
        o_rates = []
        for k in range(len(self.actors)):
            o_spikes = self.actors[k].forward(state_in, temp_count)
            o_rates.append(o_spikes)
        o_rates = np.array(o_rates)
        action_rates = np.zeros(5)
        for k in range(5):
            action_rates[k] = sum(o_rates[np.where(o_rates[:,k]==1),k][0])
        action_index = np.argmax(action_rates)
        return action_index
    
    def run_actor_critic(self, num_episodes, features='fourier'):
        last_val_best = -100
        # TRAINING ========================
        rewards = []
        #theta = np.random.rand(self.num_states)
        #theta = np.zeros(self.num_states)
        theta = np.zeros(int(math.pow(self.order+1, self.num_states)))
        w_v = np.zeros(int(math.pow(self.order+1, self.num_states)))
        alpha = 0.001
        for i in range(num_episodes):
            train_epi_i = i
            #if i > 500:
            #    self.alpha = 0.001
            #state = np.zeros(4)
            self.env.init_train()
            state = self.env.get_train_observation()
            # state = self.cartpole.reset()
            e_theta = np.zeros_like(theta)
            e_v = np.zeros(int(math.pow(self.order+1, self.num_states)))
            rt = 1; gamma = 1
            count = 0
            sigma = 1
            # while abs(state[0]) < 3 and abs(state[2]) < math.pi/2 and abs(state[3]) < math.pi and count < 1010:
            for step_num_in_episode in range(self.env.max_step_num):    # NEW EPISODE LENGTH
                # Act using actor
                fourier_state = self.fourier_feature_state(state, features)
                state_param = np.dot(theta, fourier_state)
                o_rates = []
                for k in range(len(self.actors)):
                    o_spikes = self.actors[k].forward(state, count)
                    o_rates.append(o_spikes)
                o_rates = np.array(o_rates)
                action_rates = np.zeros(5)
                for k in range(5):
                    action_rates[k] = sum(o_rates[np.where(o_rates[:,k]==1),k][0])
                action_index = np.argmax(action_rates)
                # EPSILON GREEDY ACTION SELECTION
                    # action = self.e_greedy_action(action_index)
                if random.random() < self.epsilon:
                    action = random.randint(0, 4)
                else:
                    action = action_index
                reward, observation_next, performance_list = self.env.make_action(action)
                new_state, reward, done = observation_next, reward, self.env.done_signal
                fourier_state = self.fourier_feature_state(state, features)
                fourier_new_state = self.fourier_feature_state(new_state, features)

                # Critic update
                e_v = gamma*self.lda*e_v + fourier_state
                v_s = np.dot(w_v, fourier_state)
                v_ns = np.dot(w_v, fourier_new_state)
                delta_t = rt + gamma*v_ns - v_s
                w_v += alpha*delta_t*e_v

                # Actor update
                for k in range(len(self.actors)):
                    self.actors[k].update_weights(delta_t, state, action, np.mean(rewards[-10:]))
                if self.env.done_signal == True:
                    break

                state = new_state
                count += 1
            # TRAIN SUMMARY
            log_text(File, 'train', '%d, %8.6f' % (train_epi_i, performance_list[0]), onscreen=True)
            # print("Reward after %s episodes: %s" %(i, count))
            rewards.append(count)
            # VALIDATION ===========================
            if train_epi_i % val_freq == (val_freq - 1):
                val_preformance_list = []
                for val_epi_i in range(val_num):
                    self.env.init_val()
                    for val_step_i in range(self.env.max_step_num):
                        observation = self.env.get_val_observation()
                        action_index = self.get_action_from_state(observation, temp_count=val_step_i)
                        reward, observation_next, performance_list_val = self.env.make_action(action_index)
                        new_state, reward, done = observation_next, reward, self.env.done_signal
                        if self.env.done_signal == True:
                            break
                    val_preformance_list.append(performance_list_val[0])
                val_performance_mean = sum(val_preformance_list) / len(val_preformance_list)
                if last_val_best <= val_performance_mean:
                    self.save_model(EXP_NAME + '_best')
                    log_text(File, 'val_save', '%d,   %8.6f' % (train_epi_i, val_performance_mean))
                    last_val_best = val_performance_mean
                log_text(File, 'val', '%d,   %8.6f,   %8.6f' % (train_epi_i, val_performance_mean, performance_list[0]))
        # TEST =======================
        # >>>> Test GYMIP / GYMDIP Env
        self.load_model(EXP_NAME + '_best')
        noise_type_list = ['length', 'thick', 'union']
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
                    self.env.init_test(variation_type=noise_type, variation_param=noise_param)
                    for test_step_i in range(self.env.max_step_num):
                        observation = self.env.get_test_observation()
                        action_index = self.get_action_from_state(observation, temp_count=val_step_i)
                        reward, observation_next, test_other_step_record = self.env.make_action(action_index)
                        new_state, reward, done = observation_next, reward, self.env.done_signal
                        if self.env.done_signal == True:
                            break
                    test_preformance_list.append(test_other_step_record[0])
                test_performance_mean = sum(test_preformance_list) / len(test_preformance_list)
                log_text(File, 'e_gymip', '%s,  %8.6f,   %8.6f' % (noise_type, noise_param, test_performance_mean))
                File.flush()
        # TEST END ===================
        return rewards


class SpikingActor():
    def __init__(self):
        self.inputs = 4
        self.hidden = args.hidden_group_size
        self.outputs = 5
        self.ih_weights = 0.01*np.random.rand(5, self.hidden, self.inputs)
        self.ih_bias = np.random.rand(self.hidden)
        self.ho_weights = 0.01*np.random.rand(self.outputs, self.hidden)
        self.ho_bias = np.random.rand(self.outputs)
        self.alpha = 0.001
        self.h_spikes = np.ones(self.hidden)
        self.o_spikes = np.ones(self.outputs)
        self.in_spikes = np.ones(self.inputs)
        self.hz = np.zeros(self.hidden)
        self.oz = np.zeros(self.outputs)

    def input_coding(self, state):
        maps = list(itertools.combinations(range(int(self.inputs*0.25)), r=int(self.inputs*0.25*0.5)))
        state_code = -1*np.ones(self.inputs)
        xb = int(self.inputs*0.25*(state[0] + 3)/6)
        vb = int(self.inputs*0.25*(state[1] + 10)/20)
        thetab = int(self.inputs*0.25*(state[0] + math.pi/2)/math.pi)
        omegab = int(self.inputs*0.25*(state[1] + math.pi)/(2*math.pi))
        state_code[list(maps[xb])] = 1
        state_code[list(np.array((maps[vb])) + int(self.inputs*0.25))] = 1
        state_code[list(np.array((maps[thetab])) + int(self.inputs*0.5))] = 1
        state_code[list(np.array((maps[omegab])) + int(self.inputs*0.75))] = 1
        return state_code


    def forward(self,state,count):
        inputs = state
        self.in_spikes = state

        self.hz = np.zeros((5, self.hidden))
        self.h_spikes = np.ones((5, self.hidden))
        for i in range(5):
            z = np.matmul(self.ih_weights[i], inputs)
            p = 1/(1 + np.exp(-2*z))
            self.h_spikes[i] = (p > np.random.rand(self.hidden)).astype(int)
            self.h_spikes[i] = 2*self.h_spikes[i] - 1
            self.hz[i] = 1 + np.exp(2*z*self.h_spikes[i])


        self.oz = np.zeros(self.outputs)
        self.o_spikes = np.ones(self.outputs)

        for i in range(5):
            zo = np.dot(self.ho_weights[i], self.h_spikes[i])
            po = 1/(1 + np.exp(-2*zo))
            self.o_spikes[i] = (po > np.random.rand(1)).astype(int)
            self.o_spikes[i] = 2*self.o_spikes[i] - 1
            self.oz[i] = 1 + np.exp(2*zo*self.o_spikes[i])

        return self.o_spikes

    def update_weights(self, tderror, state, action, mean_reward):
        if mean_reward > 70 and mean_reward < 190:
            self.alpha = 0.00001
        elif mean_reward > 190:
            self.alpha = 0.00001
        else:
            self.alpha = 0.001

        for i in range(5):
            if i == action:
                self.ih_weights[i] += self.alpha*tderror*np.outer(2*self.h_spikes[i]/self.hz[i], self.in_spikes)
            else:
                if self.o_spikes[i] == 1:
                    self.ih_weights[i] -= self.alpha*tderror*np.outer(2*self.h_spikes[i]/self.hz[i], self.in_spikes)
                else:
                    self.ih_weights[i] += self.alpha*tderror*np.outer(2*self.h_spikes[i]/self.hz[i], self.in_spikes)

        for i in range(5):
            if i == action:
                self.ho_weights[i] += self.alpha*tderror*np.multiply(2*self.o_spikes[i]/self.oz[i], self.h_spikes[i])
            else:
                if self.o_spikes[i] == 1:
                    self.ho_weights[i] -= self.alpha*tderror*np.multiply(2*self.o_spikes[i]/self.oz[i], self.h_spikes[i])
                else:
                    self.ho_weights[i] += self.alpha*tderror*np.multiply(2*self.o_spikes[i]/self.oz[i], self.h_spikes[i])



        

if __name__ == "__main__":
    # ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', dest='algorithm', default='sarsa')
    parser.add_argument('--features', dest='features', default='fourier')
    parser.add_argument('--selection', dest='selection', default='egreedy')
    parser.add_argument('--num_trials', dest='num_trials', default=1)
    parser.add_argument('--num_episodes', dest='num_episodes', default=3000)
    parser.add_argument('--plot', dest='plot', action='store_true')

    parser.add_argument('--rep', type=float, default=51)
    parser.add_argument('--hidden_group_size', type=int, default=23)
    parser.add_argument('--epsilon', type=float, default=0.2)
    args = parser.parse_args()

    # SETTINGS
    EXP_NAME = 'baseline3_hid%d_eps_%3.3f_rep%02d' % (args.hidden_group_size, args.epsilon, args.rep)
    val_freq, val_num, test_num = 100, 10, 10   # FOR GYMIP
    # LOG TEXT FILE
    log_text_flush_time = time.time()
    if not os.path.exists('./log_text/'):
        os.mkdir('./log_text/')
    if not os.path.exists('./log_model/'):
        os.mkdir('./log_model/')
    log_filename = './log_text/log_' + EXP_NAME + '.txt'
    File = open(log_filename, 'w')
    log_text(File, 'init', str(datetime.datetime.now()))
    log_text(File, 'arguments', str(args))


    rewards_trials = []

    step_size = 0.001 # Sarsa, fourier 0.001
    epsilon = args.epsilon
    

    for i in range(int(args.num_trials)):
        print('Trial:', i)
        td_cp = ActorCritic(order=5, epsilon=epsilon, step_size=step_size, num_states=4)
        rewards = td_cp.run_actor_critic(int(args.num_episodes), features='fourier')
        rewards_trials.append(rewards)

