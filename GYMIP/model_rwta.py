# -*- coding: utf-8 -*-
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# Model RWTA


def print_info(input_string=''):
    print('\033[94mMODEL_RWTA_INFO|\033[0m', input_string)


class RWTAprob:
    def __init__(self,
                input_size=784, output_size=10,
                hid_num=250, hid_size=4,
                remove_connection_pattern='none',       # 'none', 'hh', 'sa', 'hhsa', 'ha', 'sh'
                optimizer_name='rmsprop', optimizer_learning_rate=0.01,
                entropy_ratio=0.0, inference_noise=0.02,
                device=torch.device('cpu')):
        # Hyperparameters
        self.dim_state = input_size
        self.num_hidden = hid_num
        self.dim_hidden = hid_size
        self.dim_action = output_size
        self.infer_noise = inference_noise
        self.entropy_ratio = entropy_ratio
        # Settings
        self.remove_connection_pattern = remove_connection_pattern
        self.dev = device
        # Shape Variables
        self.dim_h = self.num_hidden * self.dim_hidden      # h, a, s: total number of neurons
        self.dim_a = 1 * self.dim_action                    # num_action = 1
        self.dim_s = self.dim_state
        self.dim_ha = self.dim_h + self.dim_a               # accumulated shape
        self.dim_has = self.dim_h + self.dim_a + self.dim_s # accumulated shape
        # Network parameters
        self.weight = torch.zeros([self.dim_has, self.dim_ha], dtype=torch.float32, device=self.dev)
        self.index_1x, self.index_1y = torch.triu_indices(self.dim_ha, self.dim_ha, device=self.dev)
        self.index_2x, self.index_2y = torch.tril_indices(self.dim_ha, self.dim_ha, device=self.dev)
        self.weight[self.index_2x, self.index_2y] = self.weight[self.index_1x, self.index_1y]
        self.bias = torch.zeros([self.dim_ha], dtype=torch.float32, device=self.dev)
        self.mask_weight = torch.ones([self.dim_has, self.dim_ha], device=self.dev)
        self.init_mask()
        # Optimizer
        self.init_optimizer(optimizer_name, optimizer_learning_rate)
        print_info('RWTA Prob initialize')
        self.print_parameter_num()
   
    def __call__(self, x):
        return self.forward(x)

    def init_mask(self):
        # Remove connections
        # >> h_i - h_i
        for index in range(0, self.dim_h, self.dim_hidden):
            self.mask_weight[index:(index + self.dim_hidden), index:(index + self.dim_hidden)] = 0
        # >> a - a
        self.mask_weight[self.dim_h:self.dim_ha, self.dim_h:self.dim_ha] = 0
        # Connection Removal in Training
        if self.remove_connection_pattern == 'none':
            pass
        elif self.remove_connection_pattern == 'sa':
            self.mask_weight[self.dim_ha:self.dim_has, self.dim_h:self.dim_ha] = 0
        elif self.remove_connection_pattern == 'hh':
            self.mask_weight[0:self.dim_h, 0:self.dim_h] = 0
        elif self.remove_connection_pattern == 'hhsa':
            self.mask_weight[0:self.dim_h, 0:self.dim_h] = 0
            self.mask_weight[self.dim_ha:self.dim_has, self.dim_h:self.dim_ha] = 0
        elif self.remove_connection_pattern == 'ha':
            self.mask_weight[0:self.dim_h, self.dim_h:self.dim_ha] = 0
            self.mask_weight[self.dim_h:self.dim_ha, 0:self.dim_h] = 0
        elif self.remove_connection_pattern == 'sh':
            self.mask_weight[self.dim_ha:self.dim_has, 0:self.dim_h] = 0
        else:
            print_info('Error in remove_connection_pattern')
        # Apply Mask
        self.weight = self.weight * self.mask_weight

    def init_optimizer(self, optimizer_name, optimizer_learning_rate):
        if optimizer_name == 'sgd':
            pass
        elif optimizer_name == 'adam':
            self.w_m = torch.zeros([self.dim_has, self.dim_ha], dtype=torch.float32, device=self.dev)
            self.w_v = torch.zeros([self.dim_has, self.dim_ha], dtype=torch.float32, device=self.dev)
            self.w_eps = 1e-8 * torch.ones([self.dim_has, self.dim_ha], dtype=torch.float32, device=self.dev)
            self.b_m = torch.zeros([self.dim_ha], dtype=torch.float32, device=self.dev)
            self.b_v = torch.zeros([self.dim_ha], dtype=torch.float32, device=self.dev)
            self.b_eps = 1e-8 * torch.ones([self.dim_ha], dtype=torch.float32, device=self.dev)
            self.beta_1 = 0.9
            self.beta_2 = 0.999
            self.beta_1_update = self.beta_1
            self.beta_2_update = self.beta_2
        elif optimizer_name == 'rmsprop':
            self.alpha = 0.99
            self.w_eps = 1e-8
            self.b_eps = 1e-8
            self.w_v = torch.zeros([self.dim_has, self.dim_ha], dtype=torch.float32, device=self.dev)
            self.b_v = torch.zeros([self.dim_ha], dtype=torch.float32, device=self.dev)
        else:
            print_info('Error in optimizer name')
        self.optimizer_name = optimizer_name
        self.optimizer_learning_rate = optimizer_learning_rate

    def print_parameter_num(self):
        part_ha = torch.sum(self.mask_weight[0:self.dim_ha, 0:self.dim_ha]) / 2
        part_s = torch.sum(self.mask_weight[self.dim_ha:self.dim_has, 0:self.dim_ha])
        print_info('weight %d' % (part_ha + part_s))
        print_info('bias %d' % (self.bias.shape[0]))

    def hid_act_softmax(self, q_hid_act):
        input_batch_size = q_hid_act.shape[0]
        # Hidden
        q_hid = q_hid_act[:, 0:self.dim_h].reshape([input_batch_size, self.num_hidden, self.dim_hidden])
        q_hid_reshape_soft = F.softmax(q_hid, dim=2)
        q_hid_soft = q_hid_reshape_soft.reshape([input_batch_size, self.dim_h])
        # Action
        q_act = q_hid_act[:, self.dim_h:self.dim_ha].reshape([input_batch_size, 1, self.dim_action])    # num_action = 1
        q_act_reshape_soft = F.softmax(q_act, dim=2)
        q_act_soft = q_act_reshape_soft.reshape([input_batch_size, self.dim_a])
        # Combine
        q_hid_act_soft = torch.cat([q_hid_soft, q_act_soft], dim=1)
        return q_hid_act_soft

    def forward(self, x):
        input_batch_size = x.shape[0]
        # Initialize
        q_ha = self.hid_act_softmax(torch.rand([input_batch_size, self.dim_ha], dtype=torch.float32, device=self.dev))
        q_s = x.clone()
        for iter_i in range(50):
            q_has = torch.cat([q_ha, q_s], dim=1)
            q_has = torch.clamp(q_has + self.infer_noise * torch.randn(q_has.size(), device=self.dev), 0, 1)
            q_has[:, 0:self.dim_ha] = q_has[:, 0:self.dim_ha] * 2
            temp_q_ha = torch.mm(q_has, self.weight) + self.bias.expand([input_batch_size, self.dim_ha])
            target_q_ha = self.hid_act_softmax(temp_q_ha)
            if torch.mean(torch.abs(q_ha - target_q_ha)).item() < 0.005:
                break
            q_ha = target_q_ha
        q_has = torch.cat([q_ha, q_s], dim=1)
        # Sample Hidden
        v_hid_act = torch.zeros_like(q_ha, device=self.dev)
        q_hid = q_ha[:, 0:self.dim_h].reshape([input_batch_size, self.num_hidden, self.dim_hidden])
        hid_dist = torch.distributions.OneHotCategorical(q_hid)
        hid_sample = hid_dist.sample()
        v_hid_act[:, 0:self.dim_h] = hid_sample.reshape([input_batch_size, self.dim_h])
        # Sample Action
        q_act = torch.clone(q_ha[:, self.dim_h:self.dim_ha])
        act_dist = torch.distributions.OneHotCategorical(q_act)
        action_dist_entropy = act_dist.entropy()
        act_sample = act_dist.sample()
        act_logprob = act_dist.log_prob(act_sample)
        v_hid_act[:, self.dim_h:self.dim_ha] = act_sample
        # Output
        return [q_act, [act_sample, act_logprob, q_has, v_hid_act, action_dist_entropy]]

    def get_filename(self, name):
        file_name = './log_model/' + name + '.pt'
        return file_name

    def save_model(self, name=''):
        # Backup 1
        torch.save(self.weight, self.get_filename(name + '_w_1'))
        torch.save(self.bias, self.get_filename(name + '_b_1'))
        torch.save(self.mask_weight, self.get_filename(name + '_m_1'))
        # Backup 2
        torch.save(self.weight, self.get_filename(name + '_w_2'))
        torch.save(self.bias, self.get_filename(name + '_b_2'))
        torch.save(self.mask_weight, self.get_filename(name + '_m_2'))

    def load_model(self, name=''):
        try:
            self.weight = torch.load(self.get_filename(name + '_w_1')).to(self.dev)
            self.bias = torch.load(self.get_filename(name + '_b_1')).to(self.dev)
            self.mask_weight = torch.load(self.get_filename(name + '_m_1')).to(self.dev)
        except:
            print_info('Error: current1 model currupted.')
            self.weight = torch.load(self.get_filename(name + '_w_2')).to(self.dev)
            self.bias = torch.load(self.get_filename(name + '_b_2')).to(self.dev)
            self.mask_weight = torch.load(self.get_filename(name + '_m_2')).to(self.dev)
        print_info('load: %s' % (self.get_filename(name)))

    def learn_ppo(self, a_logprob, old_logprob, advantage, epsilon, a_entropy,
                  old_vha, old_qhas, model_output, current_other, **kwargs):
        ratio = torch.exp(a_logprob - old_logprob.detach())
        advantage_squeeze = advantage
        target_1 = ratio * advantage_squeeze
        target_2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantage_squeeze
        loss = + torch.min(target_1, target_2).mean() + self.entropy_ratio * a_entropy.mean()
        loss.backward()
        gradient = torch.sum(model_output.grad * model_output * old_vha[:, self.dim_h:self.dim_ha], dim=1)
        gradient.detach_()
        # print(model_output.grad)
        v_hid_act = old_vha
        q_has = current_other[2]
        # -----prepare-----------
        has_prob_batch = q_has
        ha_prob_batch = q_has[:, 0:self.dim_ha]
        ha_value_batch = v_hid_act
        list_len = has_prob_batch.shape[0]
        # -----method------------
        q_has_res = has_prob_batch.expand([1, list_len, self.dim_has]).permute([1, 2, 0])
        ha_temp = (ha_value_batch - ha_prob_batch).expand([1, list_len, self.dim_ha]).permute([1, 0, 2])
        weight_target = torch.bmm(q_has_res, ha_temp)
        weight_target[:, 0:self.dim_ha, :] = 1 * weight_target[:, 0:self.dim_ha, :] \
                                           + 1 * weight_target[:, 0:self.dim_ha, :].clone().permute([0, 2, 1])
        weight_target = weight_target * gradient.expand([self.dim_has, self.dim_ha, list_len]).permute([2, 0, 1])
        weight_target = torch.mean(weight_target, dim=0) * self.mask_weight
        bias_target = (ha_value_batch - ha_prob_batch) * gradient.expand([self.dim_ha, list_len]).permute([1, 0])
        bias_target = torch.mean(bias_target, dim=0)
        if self.optimizer_name == 'adam':
            # -----weight------------
            self.w_m = weight_target * (1 - self.beta_1) + self.w_m * self.beta_1
            self.w_v = torch.square(weight_target) * (1 - self.beta_2) + self.w_v * self.beta_2
            w_m_hat = self.w_m / (1 - self.beta_1_update)
            w_v_hat = self.w_v / (1 - self.beta_2_update)
            self.weight = self.weight + w_m_hat * self.optimizer_learning_rate / (torch.sqrt(w_v_hat) + self.w_eps)
            # -----bias--------------
            self.b_m = bias_target * (1 - self.beta_1) + self.b_m * self.beta_1
            self.b_v = torch.square(bias_target) * (1 - self.beta_2) + self.b_v * self.beta_2
            b_m_hat = self.b_m / (1 - self.beta_1_update)
            b_v_hat = self.b_v / (1 - self.beta_2_update)
            self.bias = self.bias + b_m_hat * self.optimizer_learning_rate / (torch.sqrt(b_v_hat) + self.b_eps)
            # -----beta update-------
            self.beta_1_update = self.beta_1_update * self.beta_1
            self.beta_2_update = self.beta_2_update * self.beta_2
        elif self.optimizer_name == 'rmsprop':
            self.w_v = self.alpha * self.w_v + (1 - self.alpha) * torch.square(weight_target)
            self.weight = self.weight + weight_target * self.optimizer_learning_rate / (torch.sqrt(self.w_v) + self.w_eps)
            self.b_v = self.alpha * self.b_v + (1 - self.alpha) * torch.square(bias_target)
            self.bias = self.bias + bias_target * self.optimizer_learning_rate / (torch.sqrt(self.b_v) + self.b_eps)
        else:                                   # 'sgd'
            self.weight = self.weight + weight_target * self.optimizer_learning_rate
            self.bias = self.bias + bias_target * self.optimizer_learning_rate

    def learn_reinforce(self, a_logprob, advantage, a_entropy, v_ha, q_has, model_output, **kwargs):
        loss = +(a_logprob * advantage).mean() + self.entropy_ratio * a_entropy.mean()
        loss.backward()
        gradient = torch.sum(model_output.grad * model_output * v_ha[:, self.dim_h:self.dim_ha], dim=1)
        gradient.detach_()
        
        v_hid_act = v_ha
        q_has = q_has
        # -----prepare-----------
        has_prob_batch = q_has
        ha_prob_batch = q_has[:, 0:self.dim_ha]
        ha_value_batch = v_hid_act
        list_len = has_prob_batch.shape[0]
        # -----method------------
        q_has_res = has_prob_batch.expand([1, list_len, self.dim_has]).permute([1, 2, 0])
        ha_temp = (ha_value_batch - ha_prob_batch).expand([1, list_len, self.dim_ha]).permute([1, 0, 2])
        weight_target = torch.bmm(q_has_res, ha_temp)
        weight_target[:, 0:self.dim_ha, :] = 1 * weight_target[:, 0:self.dim_ha, :] \
                                           + 1 * weight_target[:, 0:self.dim_ha, :].clone().permute([0, 2, 1])
        weight_target = weight_target * gradient.expand([self.dim_has, self.dim_ha, list_len]).permute([2, 0, 1])
        weight_target = torch.mean(weight_target, dim=0) * self.mask_weight
        bias_target = (ha_value_batch - ha_prob_batch) * gradient.expand([self.dim_ha, list_len]).permute([1, 0])
        bias_target = torch.mean(bias_target, dim=0)
        if self.optimizer_name == 'adam':
            # -----weight------------
            self.w_m = weight_target * (1 - self.beta_1) + self.w_m * self.beta_1
            self.w_v = torch.square(weight_target) * (1 - self.beta_2) + self.w_v * self.beta_2
            w_m_hat = self.w_m / (1 - self.beta_1_update)
            w_v_hat = self.w_v / (1 - self.beta_2_update)
            self.weight = self.weight + w_m_hat * self.optimizer_learning_rate / (torch.sqrt(w_v_hat) + self.w_eps)
            # -----bias--------------
            self.b_m = bias_target * (1 - self.beta_1) + self.b_m * self.beta_1
            self.b_v = torch.square(bias_target) * (1 - self.beta_2) + self.b_v * self.beta_2
            b_m_hat = self.b_m / (1 - self.beta_1_update)
            b_v_hat = self.b_v / (1 - self.beta_2_update)
            self.bias = self.bias + b_m_hat * self.optimizer_learning_rate / (torch.sqrt(b_v_hat) + self.b_eps)
            # -----beta update-------
            self.beta_1_update = self.beta_1_update * self.beta_1
            self.beta_2_update = self.beta_2_update * self.beta_2
        elif self.optimizer_name == 'rmsprop':
            self.w_v = self.alpha * self.w_v + (1 - self.alpha) * torch.square(weight_target)
            self.weight = self.weight + weight_target * self.optimizer_learning_rate / (torch.sqrt(self.w_v) + self.w_eps)
            self.b_v = self.alpha * self.b_v + (1 - self.alpha) * torch.square(bias_target)
            self.bias = self.bias + bias_target * self.optimizer_learning_rate / (torch.sqrt(self.b_v) + self.b_eps)
        else:                                   # 'sgd'
            self.weight = self.weight + weight_target * self.optimizer_learning_rate
            self.bias = self.bias + bias_target * self.optimizer_learning_rate

    def add_noise_abs(self, noise_type, noise_param):
        if not noise_type in ['gaussian', 'uniform']:
            print_info('Error network noise type')
        # -----generate noise----
        if noise_type == 'gaussian':
            weight_noise = torch.randn(self.weight.size(), device=self.dev) * noise_param
            bias_noise = torch.randn(self.bias.size(), device=self.dev) * noise_param
        if noise_type == 'uniform':
            weight_noise = (torch.rand(self.weight.size(), device=self.dev) - 0.5) * 2 * noise_param
            bias_noise = (torch.rand(self.bias.size(), device=self.dev) - 0.5) * 2 * noise_param
        # -----add noise---------
        weight_noise[self.index_2x, self.index_2y] = weight_noise[self.index_1x, self.index_1y]
        self.weight += weight_noise * self.mask_weight
        self.bias += bias_noise

    def add_noise_relative(self, noise_type, noise_param):
        if not noise_type in ['gaussian', 'uniform']:
            print_info('Error network noise type')
        # initialize
        noise_rel_weight = torch.zeros_like(self.weight)
        noise_rel_bias = torch.zeros_like(self.bias)
        # HH
        if self.remove_connection_pattern in ['hh', 'hhsa']:
            pass
        else:
            mean_value, value_number = self.calc_mean_value(0, self.dim_h, 0, self.dim_h)
            noise_rel_weight[0:self.dim_h, 0:self.dim_h] = mean_value
        # HA
        if self.remove_connection_pattern in ['ha']:
            pass
        else:
            mean_value, value_number = self.calc_mean_value(self.dim_h, self.dim_ha, 0, self.dim_h)
            noise_rel_weight[self.dim_h:self.dim_ha, 0:self.dim_h] = mean_value
            noise_rel_weight[0:self.dim_h, self.dim_h:self.dim_ha] = mean_value
        # HS
        if self.remove_connection_pattern in ['sh']:
            pass
        else:
            mean_value, value_number = self.calc_mean_value(self.dim_ha, self.dim_has, 0, self.dim_h)
            noise_rel_weight[self.dim_ha:self.dim_has, 0:self.dim_h] = mean_value
        # SA
        if self.remove_connection_pattern in ['sa', 'hhsa']:
            pass
        else:
            mean_value, value_number = self.calc_mean_value(self.dim_ha, self.dim_has, self.dim_h, self.dim_ha)
            noise_rel_weight[self.dim_ha:self.dim_has, self.dim_h:self.dim_ha] = mean_value
        # bias
        mean_value = np.mean(np.abs(self.bias.cpu().numpy())).item()
        noise_rel_bias[:] = mean_value
        # generate noise
        if noise_type == 'gaussian':
            weight_noise = torch.randn(self.weight.size(), device=self.dev) * noise_param * noise_rel_weight
            bias_noise = torch.randn(self.bias.size(), device=self.dev) * noise_param * noise_rel_bias
        else:  # elif net_noise_type == 'uniform':
            weight_noise = (torch.rand(self.weight.size(), device=self.dev) - 0.5) * 2 * noise_param * noise_rel_weight
            bias_noise = (torch.rand(self.bias.size(), device=self.dev) - 0.5) * 2 * noise_param * noise_rel_bias
        # add noise
        weight_noise[self.index_2x, self.index_2y] = weight_noise[self.index_1x, self.index_1y]
        self.weight = (self.weight + weight_noise) * self.mask_weight
        self.bias = self.bias + bias_noise

    def calc_mean_value(self, p1, p2, p3, p4):
        mask_sel = np.zeros([self.dim_has, self.dim_ha], dtype=np.bool_)
        mask_sel[p1:p2, p3:p4] = 1
        weight_mask_sel = mask_sel * self.mask_weight.cpu().numpy().astype(np.bool_)
        weight_sel = self.weight.cpu().numpy()[weight_mask_sel]
        mean_value = np.mean(np.abs(weight_sel))
        value_number = weight_sel.shape[0]
        return mean_value.item(), value_number

    def random_remove_weight(self, noise_type, noise_param):
        if not noise_type in ['sh', 'hh', 'sa', 'ha']:
            print_info('Error noise type -- remove weight')
            return 1
        if noise_type == 'sh':
            temp_mask = torch.clone(self.mask_weight[self.dim_ha:self.dim_has, 0:self.dim_h])
            indices = (temp_mask == 1).nonzero()
            number_org = indices.shape[0]
            if number_org <= 0:
                print_info('Error in mask_weight')
                return 1
            number_remove = int(number_org * noise_param)
            if number_remove > 0:
                list = torch.randperm(number_org)[0:number_remove]
                for remove_i in range(number_remove):
                    ord = indices[list[remove_i]]
                    temp_mask[ord[0], ord[1]] = 0
                self.mask_weight[self.dim_ha:self.dim_has, 0:self.dim_h] = temp_mask
            print(((self.mask_weight[self.dim_ha:self.dim_has, 0:self.dim_h] == 1).nonzero().shape[0])/number_org)
        elif noise_type == 'sa':
            temp_mask = torch.clone(self.mask_weight[self.dim_ha:self.dim_has, self.dim_h:self.dim_ha])
            indices = (temp_mask == 1).nonzero()
            number_org = indices.shape[0]
            if number_org <= 0:
                print_info('Error in mask_weight')
                return 1
            number_remove = int(number_org * noise_param)
            if number_remove > 0:
                list = torch.randperm(number_org)[0:number_remove]
                for remove_i in range(number_remove):
                    ord = indices[list[remove_i]]
                    temp_mask[ord[0], ord[1]] = 0
                self.mask_weight[self.dim_ha:self.dim_has, self.dim_h:self.dim_ha] = temp_mask
            print(((self.mask_weight[self.dim_ha:self.dim_has, self.dim_h:self.dim_ha] == 1).nonzero().shape[0])/number_org)
        elif noise_type == 'hh':
            index_1x_h, index_1y_h = torch.triu_indices(self.dim_h, self.dim_h, device=self.dev)
            temp_mask = self.mask_weight[index_1x_h, index_1y_h]
            indices = (temp_mask == 1).nonzero()
            number_org = indices.shape[0]
            if number_org <= 0:
                print_info('Error in mask_weight')
                return 1
            number_remove = int(number_org * noise_param)
            if number_remove > 0:
                list = torch.randperm(number_org)[0:number_remove]
                for remove_i in range(number_remove):
                    ord = indices[list[remove_i]]
                    ord_1, ord_2 = index_1x_h[ord], index_1y_h[ord]
                    self.mask_weight[ord_1, ord_2] = 0
                    self.mask_weight[ord_2, ord_1] = 0
            print(((self.mask_weight[index_1x_h, index_1y_h] == 1).nonzero().shape[0])/number_org)
        elif noise_type == 'ha':
            temp_mask = torch.clone(self.mask_weight[0:self.dim_h, self.dim_h:self.dim_ha])
            indices = (temp_mask == 1).nonzero()
            number_org = indices.shape[0]
            if number_org <= 0:
                print_info('Error in mask_weight')
                return 1
            number_remove = int(number_org * noise_param)
            if number_remove > 0:
                list = torch.randperm(number_org)[0:number_remove]
                for remove_i in range(number_remove):
                    ord = indices[list[remove_i]]
                    temp_mask[ord[0], ord[1]] = 0
                self.mask_weight[0:self.dim_h, self.dim_h:self.dim_ha] = temp_mask
                self.mask_weight[self.dim_h:self.dim_ha, 0:self.dim_h] = torch.transpose(temp_mask, 0, 1)
            print(((self.mask_weight[self.dim_ha:self.dim_has, self.dim_h:self.dim_ha] == 1).nonzero().shape[0])/number_org)
        self.weight = self.weight * self.mask_weight


class RWTAspike(RWTAprob):
    def __init__(self, input_size=784, output_size=10,
                 hid_num=250, hid_size=4,
                 remove_connection_pattern='none',
                 optimizer_name='rmsprop', optimizer_learning_rate=0.01,
                 entropy_ratio=0, inference_noise=0.02, device=torch.device('cpu'),
                 spk_response_window='uni', spk_full_time=200, spk_resp_time=50,
                 ):
        super().__init__(input_size, output_size, hid_num, hid_size,
                         remove_connection_pattern,
                         optimizer_name, optimizer_learning_rate,
                         entropy_ratio, inference_noise, device)
        self.spk_response_window = spk_response_window
        self.spk_full_time = spk_full_time
        self.spk_resp_time = spk_resp_time
        if self.spk_response_window == 'uni':
            self.spike_response = torch.ones([1, self.spk_resp_time], dtype=torch.float32).to(self.dev) \
                                  / self.spk_resp_time
        elif self.spk_response_window == 'dexp':
            series = torch.arange(self.spk_resp_time, dtype=torch.float32).to(self.dev)
            self.spike_response = torch.exp(series / (-8.0)) - torch.exp(series / (-2.0))
            self.spike_response = self.spike_response / torch.sum(self.spike_response, dim=0)
        else:
            print_info('Error in self.spk_response_window string.')
        self.STDP_start_time, self.STDP_end_time = int(self.spk_full_time * 0.7), self.spk_full_time

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        input_batch_size = x.shape[0]
        # SPIKE MODE
        # Initialize neuron states
        q_s = x.clone()
        q_ha = self.hid_act_softmax(torch.rand([input_batch_size, self.dim_ha], dtype=torch.float32, device=self.dev))
        q_has = torch.cat([q_ha, q_s], dim=1)
        spike_record = torch.zeros([input_batch_size, self.dim_has, self.spk_full_time+self.spk_resp_time]).to(self.dev)
        prob_record = torch.zeros([input_batch_size, self.dim_has, self.spk_full_time]).to(self.dev)
        for spike_i in range(self.spk_full_time):
            # Get sample
            v_ha = torch.zeros_like(q_ha).to(self.dev)
            q_hid_reshape = torch.clone(q_ha[:, 0:self.dim_h]).reshape([input_batch_size, self.num_hidden, self.dim_hidden])
            _distribution = torch.distributions.OneHotCategorical(q_hid_reshape)
            q_hid_sample = _distribution.sample().reshape([input_batch_size, self.dim_h])
            v_ha[:, 0:self.dim_h] = q_hid_sample
            q_act_reshape = torch.clone(q_ha[:, self.dim_h:self.dim_ha])
            _distribution = torch.distributions.OneHotCategorical(q_act_reshape)
            q_act_sample = _distribution.sample()
            v_ha[:, self.dim_h:self.dim_ha] = q_act_sample
            v_ha_resp = v_ha.expand([self.spk_resp_time, input_batch_size, self.dim_ha]).permute([1, 2, 0]).clone()
            v_s = (torch.rand_like(q_s).to(self.dev) < q_s).float().expand([self.spk_resp_time, input_batch_size, self.dim_s]).permute([1, 2, 0]).clone()
            # Record sample
            spike_record[:, 0:self.dim_ha, spike_i:(spike_i+self.spk_resp_time)] += v_ha_resp \
                                    * self.spike_response.expand([input_batch_size, self.dim_ha, self.spk_resp_time])
            spike_record[:, self.dim_ha:self.dim_has, spike_i:(spike_i+self.spk_resp_time)] += v_s \
                                    * self.spike_response.expand([input_batch_size, self.dim_s, self.spk_resp_time])
            prob_record[:, :, spike_i] = q_has
            # Update q_ha
            if spike_i >= (self.spk_resp_time - 1):
                q_est_has = spike_record[:, :, spike_i].clone()
                q_est_has[: 0:self.dim_ha] = q_est_has[: 0:self.dim_ha] * 2
                temp_q_ha = torch.mm(q_est_has, self.weight) + self.bias.expand([input_batch_size, self.dim_ha])
                target_q_ha = self.hid_act_softmax(temp_q_ha)
                q_ha = target_q_ha
                q_has = torch.cat([q_ha, q_s], dim=1)
        # (Use the last q_has for action generation)
        # Get Sample
        v_hid_act = torch.zeros_like(q_ha, device=self.dev)
        q_hid = q_ha[:, 0:self.dim_h].reshape([input_batch_size, self.num_hidden, self.dim_hidden])
        hid_dist = torch.distributions.OneHotCategorical(q_hid)
        hid_sample = hid_dist.sample()
        v_hid_act[:, 0:self.dim_h] = hid_sample.reshape([input_batch_size, self.dim_h])
        q_act = q_ha[:, self.dim_h:self.dim_ha]
        act_dist = torch.distributions.OneHotCategorical(q_act)
        action_dist_entropy = act_dist.entropy()
        act_sample = act_dist.sample()
        act_logprob = act_dist.log_prob(act_sample)
        v_hid_act[:, self.dim_h:self.dim_ha] = act_sample
        output = torch.clone(q_ha[:, self.dim_h:self.dim_ha])
        # Output
        return [q_act, [act_sample, act_logprob, q_has, v_hid_act, action_dist_entropy]]
    


if __name__ == "__main__":
    print_info('RWTA model start')
    # model = RWTA_Prob(input_size=784, output_size=10, hid_num=10)
    # device = torch.device('cuda:0')
    # input_array = torch.randn([1000, 784]).to(device)
    # output_array = model(input_array)
    # model.save_model('model_mlp_test')
    
    
    


print('\033[91mFINISH: model_rwta\033[0m')


