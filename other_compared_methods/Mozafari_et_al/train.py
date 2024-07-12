

# ==============================================
# IMPLEMENTATION OF
# Bio-inspired digit recognition using reward-modulated spike-timing-dependent plasticity in deep convolutional networks

# BASED ON
# https://github.com/miladmozafari/SpykeTorch
# SPECIFICALLY
# https://github.com/miladmozafari/SpykeTorch/blob/master/MozafariDeep.py

# ADAPTED TO THE MNIST TASK, WITH NETWORK SIZES AND INPUT PROCESSING CHANGED
# ADDED TESTS OF WEIGHT AND INPUT PERTURBATIONS
# ==============================================


'''
for REP in {0..4}; do
python train.py --rep $REP --thread 6;
done
'''

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.nn.parameter import Parameter
import torchvision
import numpy as np
from SpykeTorch import snn
from SpykeTorch import functional as sf
from SpykeTorch import visualization as vis
from SpykeTorch import utils
from torchvision import transforms

import math
# import utils1
import argparse
parser = argparse.ArgumentParser(description='Description: 1')
parser.add_argument('--rep', type=int, default=51)
parser.add_argument('--thread', type=int, default=-1)
args = parser.parse_args()

if args.thread == -1:
    pass
else:
    torch.set_num_threads(args.thread)
use_cuda = True
EXP_NAME = 'train4encoding_%02d' % (args.rep)
import skimage

class MozafariMNIST2018(nn.Module):
    def __init__(self):
        super(MozafariMNIST2018, self).__init__()

        self.conv1 = snn.Convolution(1, 500, 28, 0.8, 0.05)
        self.conv1_t = 15 * (28*28/(5*5)) * 0.35 * 0.16
        self.k1 = 16
        self.r1 = 0

        self.conv3 = snn.Convolution(500, 10, 1, 0.8, 0.05)

        self.stdp1 = snn.STDP(self.conv1, (0.004, -0.003))
        self.stdp3 = snn.STDP(self.conv3, (0.004, -0.003), False, 0.2, 0.8)
        self.anti_stdp3 = snn.STDP(self.conv3, (-0.004, 0.0005), False, 0.2, 0.8)
        self.max_ap = Parameter(torch.Tensor([0.15]))

        self.decision_map = []
        for i in range(10):
            self.decision_map.extend([i]*1)

        self.ctx = {"input_spikes":None, "potentials":None, "output_spikes":None, "winners":None}
        self.spk_cnt1 = 0
        self.spk_cnt2 = 0

    def forward(self, input, max_layer):
        # input = sf.pad(input.float(), (2,2,2,2), 0)
        input = input.float()
        if self.training:
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, self.conv1_t, True)
            if max_layer == 1:
                self.spk_cnt1 += 1
                if self.spk_cnt1 >= 500:
                    self.spk_cnt1 = 0
                    ap = torch.tensor(self.stdp1.learning_rate[0][0].item(), device=self.stdp1.learning_rate[0][0].device) * 2
                    ap = torch.min(ap, self.max_ap)
                    an = ap * -0.75
                    self.stdp1.update_all_learning_rate(ap.item(), an.item())
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(pot, self.k1, self.r1, spk)
                self.ctx["input_spikes"] = input
                self.ctx["potentials"] = pot
                self.ctx["output_spikes"] = spk
                self.ctx["winners"] = winners
                return spk, pot
            
            '''     LAYER 2 IS REMOVED IN THIS VERSION
            # spk_in = sf.pad(sf.pooling(spk, 2, 2), (1,1,1,1))
            spk_in = spk
            pot = self.conv2(spk_in)
            spk, pot = sf.fire(pot, self.conv2_t, True)
            if max_layer == 2:
                self.spk_cnt2 += 1
                if self.spk_cnt2 >= 500:
                    self.spk_cnt2 = 0
                    ap = torch.tensor(self.stdp2.learning_rate[0][0].item(), device=self.stdp2.learning_rate[0][0].device) * 2
                    ap = torch.min(ap, self.max_ap)
                    an = ap * -0.75
                    self.stdp2.update_all_learning_rate(ap.item(), an.item())
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(pot, self.k2, self.r2, spk)
                self.ctx["input_spikes"] = spk_in
                self.ctx["potentials"] = pot
                self.ctx["output_spikes"] = spk
                self.ctx["winners"] = winners
                return spk, pot
            '''
            
            # spk_in = sf.pad(sf.pooling(spk, 3, 3), (2,2,2,2))
            spk_in = spk
            pot = self.conv3(spk_in)
            spk = sf.fire(pot)
            winners = sf.get_k_winners(pot, 1, 0, spk)
            self.ctx["input_spikes"] = spk_in
            self.ctx["potentials"] = pot
            self.ctx["output_spikes"] = spk
            self.ctx["winners"] = winners
            output = -1
            if len(winners) != 0:
                output = self.decision_map[winners[0][0]]
            return output
        else:
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, self.conv1_t, True)
            if max_layer == 1:
                return spk, pot
            '''
            pot = self.conv2(spk)
            spk, pot = sf.fire(pot, self.conv2_t, True)
            if max_layer == 2:
                return spk, pot
            '''
            pot = self.conv3(spk)
            spk = sf.fire(pot)
            winners = sf.get_k_winners(pot, 1, 0, spk)
            output = -1
            if len(winners) != 0:
                output = self.decision_map[winners[0][0]]
            return output
    
    def stdp(self, layer_idx):
        if layer_idx == 1:
            self.stdp1(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
        if layer_idx == 2:
            self.stdp2(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

    def update_learning_rates(self, stdp_ap, stdp_an, anti_stdp_ap, anti_stdp_an):
        self.stdp3.update_all_learning_rate(stdp_ap, stdp_an)
        self.anti_stdp3.update_all_learning_rate(anti_stdp_an, anti_stdp_ap)

    def reward(self):
        self.stdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

    def punish(self):
        self.anti_stdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

def train_unsupervise(network, data, layer_idx):
    network.train()
    for i in range(len(data)):
        data_in = data[i]
        if use_cuda:
            data_in = data_in.cuda()
        network(data_in, layer_idx)
        network.stdp(layer_idx)

def train_rl(network, data, target):
    network.train()
    perf = np.array([0,0,0]) # correct, wrong, silence
    for i in range(len(data)):
        data_in = data[i]
        target_in = target[i]
        if use_cuda:
            data_in = data_in.cuda()
            target_in = target_in.cuda()
        d = network(data_in, 3)
        if d != -1:
            if d == target_in:
                perf[0]+=1
                network.reward()
            else:
                perf[1]+=1
                network.punish()
        else:
            perf[2]+=1
    return perf/len(data)

def test(network, data, target):
    network.eval()
    perf = np.array([0,0,0]) # correct, wrong, silence
    for i in range(len(data)):
        data_in = data[i]
        target_in = target[i]
        if use_cuda:
            data_in = data_in.cuda()
            target_in = target_in.cuda()
        d = network(data_in, 3)
        if d != -1:
            if d == target_in:
                perf[0]+=1
            else:
                perf[1]+=1
        else:
            perf[2]+=1
    return perf/len(data)

class S1C1Transform:
    def __init__(self, filter, timesteps = 15, noise_type=None, noise_value=None):
        self.to_tensor = transforms.ToTensor()
        self.filter = filter
        self.temporal_transform = utils.Intensity2Latency(timesteps)
        self.cnt = 0
        self.noise_type = noise_type
        self.noise_value = noise_value
    def __call__(self, image):
        if self.cnt % 1000 == 0:
            print(self.cnt)
        self.cnt+=1
        image = self.to_tensor(image)
        if self.noise_type != None:
            assert image.ndim == 3
            assert image.shape[0] == 1
            image_numpy = image.cpu().numpy()
            image_numpy = image_numpy[0, :, :]
            image_numpy_noise = self.add_noise(image_numpy)
            image_numpy_noise = np.float32(image_numpy_noise)
            image = torch.from_numpy(image_numpy_noise)
            image = torch.unsqueeze(image, dim=0)
        image = image * 255
        image.unsqueeze_(0)
        # image = self.filter(image)
        image = sf.local_normalization(image, 8)
        temporal_image = self.temporal_transform(image)
        return temporal_image.sign().byte()
    def add_noise(self, temp_img):
        noise_type = self.noise_type
        noise_param = self.noise_value
        if noise_type == 'gaussian':
            temp_img = skimage.util.random_noise(temp_img, mode='gaussian', seed=None,
                                                    clip=True, var=noise_param**2)
        elif noise_type == 'pepper':
            temp_img = skimage.util.random_noise(temp_img, mode='pepper', seed=None,
                                                    clip=True, amount=noise_param)
        elif noise_type == 'salt':
            temp_img = skimage.util.random_noise(temp_img, mode='salt', seed=None, clip=True,
                                                    amount=noise_param)
        elif noise_type == 's&p':
            temp_img = skimage.util.random_noise(temp_img, mode='s&p', seed=None, clip=True,
                                                    amount=noise_param, salt_vs_pepper=0.5)
        elif noise_type == 'gaussian&salt':
            temp_img = skimage.util.random_noise(temp_img, mode='gaussian', seed=None, clip=True,
                                                    var=0.05**2)
            temp_img = skimage.util.random_noise(temp_img, mode='salt', seed=None, clip=True,
                                                    amount=noise_param)
        return temp_img

kernels = [ utils.DoGKernel(3,3/9,6/9),
            utils.DoGKernel(3,6/9,3/9),
            utils.DoGKernel(7,7/9,14/9),
            utils.DoGKernel(7,14/9,7/9),
            utils.DoGKernel(13,13/9,26/9),
            utils.DoGKernel(13,26/9,13/9)]
filter = utils.Filter(kernels, padding = 6, thresholds = 50)
s1c1 = S1C1Transform(filter)

data_root = "data"
MNIST_train = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform = s1c1))
MNIST_test = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform = s1c1))
MNIST_loader = DataLoader(MNIST_train, batch_size=1000, shuffle=False)
MNIST_testLoader = DataLoader(MNIST_test, batch_size=len(MNIST_test), shuffle=False)

mozafari = MozafariMNIST2018()
if use_cuda:
    mozafari.cuda()

# Training The First Layer
print("Training the first layer")
if os.path.isfile('./log_model/' + EXP_NAME + "saved_l1.net"):
    mozafari.load_state_dict(torch.load('./log_model/' + EXP_NAME + "saved_l1.net"))
else:
    for epoch in range(2):
        print("Epoch", epoch)
        iter = 0
        for data,targets in MNIST_loader:
            print("Iteration", iter)
            train_unsupervise(mozafari, data, 1)
            print("Done!")
            iter+=1
    torch.save(mozafari.state_dict(), './log_model/' + EXP_NAME + "saved_l1.net")
'''
# Training The Second Layer
print("Training the second layer")
if os.path.isfile('./log_model/' + EXP_NAME + "saved_l2.net"):
    mozafari.load_state_dict(torch.load('./log_model/' + EXP_NAME + "saved_l2.net"))
else:
    for epoch in range(4):
        print("Epoch", epoch)
        iter = 0
        for data,targets in MNIST_loader:
            print("Iteration", iter)
            train_unsupervise(mozafari, data, 2)
            print("Done!")
            iter+=1
    torch.save(mozafari.state_dict(), './log_model/' + EXP_NAME + "saved_l2.net")
'''

# initial adaptive learning rates
apr = mozafari.stdp3.learning_rate[0][0].item()
anr = mozafari.stdp3.learning_rate[0][1].item()
app = mozafari.anti_stdp3.learning_rate[0][1].item()
anp = mozafari.anti_stdp3.learning_rate[0][0].item()

adaptive_min = 0
adaptive_int = 1
apr_adapt = ((1.0 - 1.0 / 10) * adaptive_int + adaptive_min) * apr
anr_adapt = ((1.0 - 1.0 / 10) * adaptive_int + adaptive_min) * anr
app_adapt = ((1.0 / 10) * adaptive_int + adaptive_min) * app
anp_adapt = ((1.0 / 10) * adaptive_int + adaptive_min) * anp

# perf
best_train = np.array([0.0,0.0,0.0,0.0]) # correct, wrong, silence, epoch
best_test = np.array([0.0,0.0,0.0,0.0]) # correct, wrong, silence, epoch

# Training The Third Layer
print("Training the third layer")

# ORIGINAL EPOCH NUMBER TOTAL == 680
for epoch in range(100):
    print("Epoch #:", epoch)
    perf_train = np.array([0.0,0.0,0.0])
    for data,targets in MNIST_loader:
        perf_train_batch = train_rl(mozafari, data, targets)
        print(perf_train_batch)
        #update adaptive learning rates
        apr_adapt = apr * (perf_train_batch[1] * adaptive_int + adaptive_min)
        anr_adapt = anr * (perf_train_batch[1] * adaptive_int + adaptive_min)
        app_adapt = app * (perf_train_batch[0] * adaptive_int + adaptive_min)
        anp_adapt = anp * (perf_train_batch[0] * adaptive_int + adaptive_min)
        mozafari.update_learning_rates(apr_adapt, anr_adapt, app_adapt, anp_adapt)
        perf_train += perf_train_batch
    perf_train /= len(MNIST_loader)
    if best_train[0] <= perf_train[0]:
        best_train = np.append(perf_train, epoch)
    print("Current Train:", perf_train)
    print("   Best Train:", best_train)

    for data,targets in MNIST_testLoader:
        perf_test = test(mozafari, data, targets)
        if best_test[0] <= perf_test[0]:
            best_test = np.append(perf_test, epoch)
            torch.save(mozafari.state_dict(), './log_model/' + EXP_NAME + "saved_l3.net")
        print(" Current Test:", perf_test)
        print("    Best Test:", best_test)



# ==========TEST LOG=========================
import time
def log_text(file_handle, type_str, record_text, onscreen=True):
    global log_text_flush_time
    if onscreen:
        print('\033[92m%s\033[0m' % (type_str).ljust(10), record_text)
    file_handle.write((type_str+',').ljust(10) + record_text + '\n')
    if time.time() - log_text_flush_time > 10:
        log_text_flush_time = time.time()
        file_handle.flush()
        os.fsync(file_handle.fileno())
log_filename = './log_text/log_' + EXP_NAME + '.txt'
File = open(log_filename, 'w')
log_text_flush_time = time.time()


# ===========================================
# Model Summary
l1_param = list(mozafari.conv1.parameters())[0]
l3_param = list(mozafari.conv3.parameters())[0]
print('Conv shapes')
print(l1_param.shape, l1_param.numel())
print(l3_param.shape, l3_param.numel())
print('Total')
print(l1_param.numel() + l3_param.numel())
print('Alternative', sum(p.numel() for p in mozafari.parameters()))

# ===========================================
# Test Weight Noise
noise_type_list = ['gaussian', 'uniform']
device = torch.device('cuda:0')
for noise_type in noise_type_list:
    # -----noise param-------
    if noise_type in ['gaussian']:
        noise_param_list = np.arange(0, 5, 0.1)
    else:    # if noise_type in ['uniform']:
        noise_param_list = np.arange(0, 5, 0.1)
    for noise_param in noise_param_list:
        # Load Model
        mozafari.load_state_dict(torch.load('./log_model/' + EXP_NAME + "saved_l3.net"))
        # Add Noise to Parameters
        with torch.no_grad():
            for param in mozafari.conv1.parameters():
                mean_value = np.mean(np.abs(param.cpu().numpy()))
                if noise_type == 'gaussian':
                    param.add_(torch.randn(param.size()).to(device) * noise_param * mean_value)
                if noise_type == 'uniform':
                    param.add_((torch.rand(param.size()).to(device) - 0.5) * 2 * noise_param * mean_value)
            '''
            for param in mozafari.conv2.parameters():
                mean_value = np.mean(np.abs(param.cpu().numpy()))
                if noise_type == 'gaussian':
                    param.add_(torch.randn(param.size()).to(device) * noise_param * mean_value)
                if noise_type == 'uniform':
                    param.add_((torch.rand(param.size()).to(device) - 0.5) * 2 * noise_param * mean_value)
            '''
            for param in mozafari.conv3.parameters():
                mean_value = np.mean(np.abs(param.cpu().numpy()))
                if noise_type == 'gaussian':
                    param.add_(torch.randn(param.size()).to(device) * noise_param * mean_value)
                if noise_type == 'uniform':
                    param.add_((torch.rand(param.size()).to(device) - 0.5) * 2 * noise_param * mean_value)
        # Test Accuracy
        for data,targets in MNIST_testLoader:
            perf_test = test(mozafari, data, targets)                
            print(" Current Test:", perf_test)
        log_text(File, 'w_rel_noise', '%s,  %8.6f,   %8.6f' % (noise_type, noise_param, perf_test[0]))
        File.flush()




# ===========================================
# Test Input Noise
print('Test input noise')
mozafari.load_state_dict(torch.load('./log_model/' + EXP_NAME + "saved_l3.net"))
noise_type_list = ['gaussian', 'pepper', 'salt', 's&p', 'gaussian&salt']
for noise_type in noise_type_list:
    if noise_type in ['gaussian']:
        noise_param_list = np.arange(0, 1.6, 0.05)
    if noise_type in ['pepper', 'salt', 's&p']:
        noise_param_list = np.arange(0, 0.51, 0.02)
    if noise_type in ['gaussian&salt']:
        noise_param_list = np.arange(0, 0.505, 0.005)
    for noise_param in noise_param_list:
        # INIT NOISE
        print('Init noise', noise_type, noise_param)
        s1c1_with_noise = S1C1Transform(filter, noise_type=noise_type, noise_value=noise_param)
        MNIST_test_noise = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform = s1c1_with_noise))
        MNIST_testLoader_noise = DataLoader(MNIST_test_noise, batch_size=len(MNIST_test), shuffle=False)
        for data,targets in MNIST_testLoader_noise:   # SINGLE EPISODE
            perf_test = test(mozafari, data, targets)
            print(" Current Test:", perf_test)
        log_text(File, 'i_noise', '%s,  %8.6f,   %8.6f' % (noise_type, noise_param, perf_test[0]))
        # File.flush()


