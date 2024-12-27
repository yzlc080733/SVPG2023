# Code for Spiking Variarional Policy Gradient (SVPG)

This is the code repository for implementing the experiments in the journal paper:

Spiking Variational Policy Gradient for Brain Inspired Reinforcement Learning

[Link to paper](https://doi.org/10.1109/TPAMI.2024.3511936)

An old version of the code is at [GitHub - yzlc080733/BMVC2022_SVPG](https://github.com/yzlc080733/BMVC2022_SVPG), which corresponds to the BMVC 2022 conference paper.

Five RL tasks are used, namely MNIST, GYMIP, DOOM, AI2THOR, and ROBOTARM.

## Set up

### Packages

Our experiments are run with `python==3.8` on Linux (Ubuntu 20.04 and Red Hat 8.7). Below is a list of packages and versions. It is possible that packages of other versions can work fine.

* Note that, when writing this document, the version requirement on `spikingjelly` and `protobuf` is strict.

* You may need to install system dependencies to make `mujoco` and `vizdoom` work.

```bash
# Basic
pip install torch==1.13 numpy==1.19 matplotlib
pip install scikit-image scikit-learn
pip install opencv-python==4.7.0 protobuf==3.20.0
# ANN2SNN related
pip install spikingjelly==0.0.0.0.8 onnx==1.8.0 onnxruntime==1.10.0
# BPTT related
pip install snntorch==0.5.3
# GYMIP related
pip install gym==0.26.2 mujoco==2.2.0
# DOOM related
pip install vizdoom==1.1.14
# AI2THOR related
pip install ai2thor
# ROBOTARM related
# NOTE -- PyRep and CoppeliaSim are needed.
#      -- Please follow https://github.com/stepjam/PyRep
```

### Dataset and environment

* We do not attach the MNIST dataset. You may use `python prepare_mnist.py` to prepare the dataset.

* For convenience in modifying the Gym InvertedPendulum environment, we included a copy of the Gym library [gym/gym at 0.26.2 · openai/gym · GitHub](https://github.com/openai/gym/tree/0.26.2/gym). If you choose to download the Gym library separately, you can use `python prepare_gymip.py` to make the modification.

## Running the code

### Notes

* ANN2SNN relies on the training results from BP. The hyperparameters in ANN2SNN is used for finding the corresponding BP checkpoints in `./log_model/`.
* Below are examples for running each method on each task.
* For GYMIP, you may add `--alg reinforce` to switch to the REINFORCE algorithm.
* For AI2THOR and ROBOTARM, an NVIDIA GPU and an X server (program) is needed. We recommend running on a Linux machine with NVIDIA GPU and a monitor attached. Running on a headless server may fail.

### MNIST

```bash
cd ./MNIST/
# SVPG-rate
python run_PPO.py --task mnist --model rwtaprob --cuda -1 --hidden_num 500 --hid_group_num 50 --hid_group_size 10 --lr 0.001 --entropy 0.2 --rwta_del_connection none --ignore_checkpoint
# SVPG-spike
python run_PPO.py --task mnist --model rwtaspk --cuda -1 --hidden_num 500 --hid_group_num 50 --hid_group_size 10 --lr 0.001 --entropy 0.2 --rwta_del_connection none --ignore_checkpoint
# BP
python run_PPO.py --task mnist --model mlp3relu --cuda -1 --hidden_num 500 --hid_group_num 50 --hid_group_size 10 --lr 0.0001 --entropy 1.0 --rwta_del_connection none --ignore_checkpoint
# BPTT
python run_PPO.py --task mnist --model snnbptt --cuda -1 --hidden_num 500 --hid_group_num 50 --hid_group_size 10 --lr 0.0001 --entropy 1.0 --rwta_del_connection none --ignore_checkpoint
# ANN2SNN
python run_PPO.py --task mnist --model ann2snn --cuda -1 --hidden_num 500 --hid_group_num 50 --hid_group_size 10 --lr 0.0001 --entropy 1.0 --rwta_del_connection none --ignore_checkpoint
```

### GYMIP

```bash
cd ./GYMIP/
# Note: In this code the network size is coded in run_RL.py
# SVPG
python run_RL.py --task gymip --model rwtaprob --cuda -1 --lr 0.001 --entropy 5 --rwta_del_connection none --ignore_checkpoint
# BP
python run_RL.py --task gymip --model mlp3relu --cuda -1 --lr 0.0001 --entropy 1 --rwta_del_connection none --ignore_checkpoint
# BPTT
python run_RL.py --task gymip --model snnbptt --cuda -1 --lr 0.0001 --entropy 0.5 --rwta_del_connection none --snn_num_steps 15 --ignore_checkpoint
# ANN2SNN
python run_RL.py --task gymip --model ann2snn --cuda -1 --lr 0.0001 --entropy 1 --rwta_del_connection none --ignore_checkpoint
```

### DOOM

```bash
cd ./DOOM/
# SVPG
python run_PPO.py --task vizdoom --model rwtaprob --cuda -1 --lr 0.0001 --entropy 0.02 --rwta_del_connection none --ignore_checkpoint
# BP
python run_PPO.py --task vizdoom --model mlp3relu --cuda -1 --lr 0.0001 --entropy 0.02 --rwta_del_connection none --ignore_checkpoint
# BPTT
python run_PPO.py --task vizdoom --model snnbptt --cuda 0 --lr 0.0001 --entropy 0.02 --rwta_del_connection none --ignore_checkpoint
# ANN2SNN
python run_ANN2SNN.py --task vizdoom --model mlp3relu --cuda 0 --lr 0.0001 --entropy 0.02 --rwta_del_connection none --ignore_checkpoint
```

### AI2THOR

Similar to DOOM. The folder is `./AI2THOR/`.

Replace `--task vizdoom` to `--task ai2thor` to train on this task.

### ROBOTARM

Similar to DOOM. The folder is `./ROBOTARM/`.

Replace `--task vizdoom` to `--task robotarm` to train on this task.

### Results

You can find the training and testing records in `./log_text/` and `./log_model/`. Some outputs of the ANN2SNN method is stored in `./ann2snn/`.
