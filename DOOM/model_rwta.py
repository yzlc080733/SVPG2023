# -*- coding: utf-8 -*-
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



# Model RWTA




def print_info(input_string=''):
    INFO_MESSAGE = '\033[94mMODEL_RWTA_INFO|\033[0m'
    print(INFO_MESSAGE, input_string)



class RWTA_Prob:
    def __init__(self,
                input_size=784, output_size=10,
                hid_num=250, hid_size=4,
                infer_noise = 0.02,
                remove_connection_pattern='none',       # 'none', 'hh', 'sa', 'hhsa'
                optimizer_name='sgd', optimizer_learning_rate=0.01,
                entropy_ratio=0.0,
                device=torch.device('cpu')):
        # -----hyper params------
        self.dim_state = input_size
        self.num_hidden = hid_num
        self.dim_hidden = hid_size
        self.dim_action = output_size
        self.infer_noise = 0.02
        self.entropy_ratio = entropy_ratio
        # -----settings----------
        self.remove_connection_pattern = remove_connection_pattern
        self.dev = device
        # -----init--------------
        self.init_shape_variables()
        self.init_parameters()
        self.init_mask()                # weight multiplication is included
        self.init_optimizer(optimizer_name, optimizer_learning_rate)
        print_info('RWTA Prob initialize')
    
    
    def init_shape_variables(self):
        self.dim_h = self.num_hidden * self.dim_hidden      # h, a, s: total number of neurons
        self.dim_a = 1 * self.dim_action                    # num_action = 1
        self.dim_s = self.dim_state
        self.dim_ha = self.dim_h + self.dim_a               # accumulated shape
        self.dim_has = self.dim_h + self.dim_a + self.dim_s # accumulated shape
    
    def init_parameters(self):
        self.weight = torch.zeros([self.dim_has, self.dim_ha], dtype=torch.float32, device=self.dev)
        self.index_1x, self.index_1y = torch.triu_indices(self.dim_ha, self.dim_ha)
        self.index_2x, self.index_2y = torch.tril_indices(self.dim_ha, self.dim_ha)
        self.weight[self.index_2x, self.index_2y] = self.weight[self.index_1x, self.index_1y]
        self.bias = torch.zeros([self.dim_ha], dtype=torch.float32, device=self.dev)
        
    def init_mask(self):
        self.mask_weight = torch.ones([self.dim_has, self.dim_ha], device=self.dev)
        # -----rm connections----
        # h_i - h_i
        for index in range(0, self.dim_h, self.dim_hidden):
            self.mask_weight[index:(index + self.dim_hidden), index:(index + self.dim_hidden)] = 0
        # a-a
        self.mask_weight[self.dim_h:self.dim_ha, self.dim_h:self.dim_ha] = 0
        # -----patterns----------           'none', 'hh', 'sa', 'hhsa'
        if self.remove_connection_pattern == 'none':
            pass
        elif self.remove_connection_pattern == 'sa':
            self.mask_weight[self.dim_ha:self.dim_has, self.dim_h:self.dim_ha] = 0
        elif self.remove_connection_pattern == 'hh':
            self.mask_weight[0:self.dim_h, 0:self.dim_h] = 0
        elif self.remove_connection_pattern == 'hhsa':
            self.mask_weight[0:self.dim_h, 0:self.dim_h] = 0
            self.mask_weight[self.dim_ha:self.dim_has, self.dim_h:self.dim_ha] = 0
        else:
            print_info('Error in remove_connection_pattern')
        # -----apply mask--------
        self.weight = self.weight * self.mask_weight

    def init_optimizer(self, optimizer_name, optimizer_learning_rate):
        if optimizer_name in ['sgd']:
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
        else:
            print_info('Error in optimizer name')
        self.optimizer_name = optimizer_name
        self.optimizer_learning_rate = optimizer_learning_rate

    def hid_act_softmax(self, q_hid_act):
        input_batch_size = q_hid_act.shape[0]
        q_hid = q_hid_act[:, 0:self.dim_h].reshape([input_batch_size, self.num_hidden, self.dim_hidden])
        q_hid_reshape_soft = F.softmax(q_hid, dim=2)
        q_hid_soft = q_hid_reshape_soft.reshape([input_batch_size, self.dim_h])
        q_act = q_hid_act[:, self.dim_h:self.dim_ha].reshape([input_batch_size, 1, self.dim_action])    # num_action = 1
        q_act_reshape_soft = F.softmax(q_act, dim=2)
        q_act_soft = q_act_reshape_soft.reshape([input_batch_size, self.dim_a])
        q_hid_act_soft = torch.cat([q_hid_soft, q_act_soft], dim=1)
        return q_hid_act_soft

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        input_batch_size = x.shape[0]
        # Get Probability
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
        return [output, [act_sample, act_logprob, q_has, v_hid_act, action_dist_entropy]]

    def calc_extra_result(self, x):
        output = self.forward(x)
        q_has = self.forward_state[0]
        hid_rate = q_has[:, 0:self.dim_h]
        return hid_rate, output

    def to(self, torch_device=torch.device('cpu')):       # replacement for torch model.to()
        return self

    def get_filename(self, name):
        file_name = './log_model/' + name + '.pt'
        return file_name

    def save_model(self, name=''):
        # -----backup 1----------
        torch.save(self.weight, self.get_filename(name + '_w_1'))
        torch.save(self.bias, self.get_filename(name + '_b_1'))
        torch.save(self.mask_weight, self.get_filename(name + '_m_1'))
        # -----backup 2----------
        torch.save(self.weight, self.get_filename(name + '_w_2'))
        torch.save(self.bias, self.get_filename(name + '_b_2'))
        torch.save(self.mask_weight, self.get_filename(name + '_m_2'))

    def load_model(self, name=''):
        try:
            self.weight = torch.load(self.get_filename(name + '_w_1'))
            self.bias = torch.load(self.get_filename(name + '_b_1'))
            self.mask_weight = torch.load(self.get_filename(name + '_m_1'))
        except:
            print_info('Error: current1 model currupted.')
            self.weight = torch.load(self.get_filename(name + '_w_2'))
            self.bias = torch.load(self.get_filename(name + '_b_2'))
            self.mask_weight = torch.load(self.get_filename(name + '_m_2'))
        print_info('load: %s' % (self.get_filename(name)))

    def learn_step(self, **kwargs):         # not needed at this stage
        pass

    def handle_other_list(self, other):
        if isinstance(other[0], list):
            content_num = len(other[0])
            reorder = []
            for c_i in range(content_num):
                content_i_list = []
                for e_i in range(len(other)):
                    content_i_list.append(other[e_i][c_i])
                if c_i in [1, 2, 3, 4, 5]:          # q_has
                    reorder.append(torch.cat(content_i_list))
                else:
                    reorder.append(content_i_list)
            return reorder
        else:
            return other

    def learn_episode_ppo(self, a_logprob, old_logprob, advantage, epsilon, a_entropy, other, model_output, a_onehot, **kwargs):
        ratio = torch.exp(a_logprob - old_logprob.detach())
        advantage_squeeze = torch.squeeze(advantage.detach(), dim=1)
        target_1 = ratio * advantage_squeeze
        target_2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantage_squeeze
        loss = + torch.min(target_1, target_2).mean() + self.entropy_ratio * a_entropy.mean()
        loss.backward()
        gradient = torch.sum(model_output.grad * model_output * a_onehot.detach(), dim=1)
        gradient.detach_()

        # [act_sample, act_log_prob, q_has, v_hid_act, action_dist_entropy] = self.handle_other_list(other[0])
        [_0, _1, _2, v_hid_act, _4] = self.handle_other_list(other[0])
        [_0, _1, q_has, _3, _4] = self.handle_other_list(other[1])
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
        # print('=======================')
        # print(weight_target.max(), gradient.max(), q_has_res.max(), ha_temp.max())
        bias_target = ha_value_batch * gradient.expand([self.dim_ha, list_len]).permute([1, 0])
        bias_target = torch.mean(bias_target, dim=0)
        # print(weight_target.max(), gradient.max(), q_has_res.max(), ha_temp.max())
        # print(weight_target.min(), gradient.min(), q_has_res.min(), ha_temp.min())
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
        else:                                   # 'sgd'
            self.weight = self.weight + weight_target * self.optimizer_learning_rate
            self.bias = self.bias + bias_target * self.optimizer_learning_rate
        # -----fake loss display-
        loss = loss
        return loss.detach().item()

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



if __name__ == "__main__":
    print_info('RWTA model start')
    model = RWTA_Prob(input_size=784, output_size=10, hid_num=10)
    device = torch.device('cuda:0')
    # input_array = torch.randn([1000, 784]).to(device)
    # output_array = model(input_array)
    # model.save_model('model_mlp_test')
    
    
    


print('\033[91mFINISH: model_mlp\033[0m')


