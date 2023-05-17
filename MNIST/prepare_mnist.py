# -*- coding: utf-8 -*-
import os
import urllib.request
import zipfile
import gzip
from urllib.request import urlretrieve
import numpy as np
import pickle

# Download dataset from https://deepai.org/dataset/mnist
# Reference: https://mattpetersen.github.io/load-mnist-with-numpy


def print_mnist_info(input_string):
    MNIST_PRE_INFO_MESSAGE = '\033[93mMNIST_PRE_INFO|\033[0m'
    print(MNIST_PRE_INFO_MESSAGE, input_string)


def download_mnist():
    mnist_folder_name = './env/mnist/'
    mnist_file_name_list = [
        'train-images-idx3-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-labels-idx1-ubyte.gz',
    ]
    check_file = True
    for filename in mnist_file_name_list:
        if not os.path.exists(mnist_folder_name + filename):
            check_file = False
    if not check_file:
        print_mnist_info('Data incomplete. Download.')
        urllib.request.urlretrieve('https://data.deepai.org/mnist.zip', 'TEMPFILE_mnist.zip')
        with zipfile.ZipFile('./TEMPFILE_mnist.zip', 'r') as zip_ref:
            zip_ref.extractall('./env/mnist')
        os.remove('TEMPFILE_mnist.zip')
    else:
        print_mnist_info('Use existing data files.')


def load_mnist():
    with gzip.open('./env/mnist/train-images-idx3-ubyte.gz') as f:
        pixels = np.frombuffer(f.read(), 'B', offset=16)
        train_i = pixels.reshape(-1, 784)
    with gzip.open('./env/mnist/t10k-images-idx3-ubyte.gz') as f:
        pixels = np.frombuffer(f.read(), 'B', offset=16)
        test_i = pixels.reshape(-1, 784)
    with gzip.open('./env/mnist/train-labels-idx1-ubyte.gz') as f:
        train_l = np.frombuffer(f.read(), 'B', offset=8)
    with gzip.open('./env/mnist/t10k-labels-idx1-ubyte.gz') as f:
        test_l = np.frombuffer(f.read(), 'B', offset=8)
    mnist_content = {
        'training_images': train_i,
        'training_labels': train_l,
        'test_images': test_i,
        'test_labels': test_l,
    }
    print_mnist_info('Data info: dtype, shape, max, min')
    for key in mnist_content:
        key_item = mnist_content[key]
        print('    ', key, key_item.dtype, key_item.shape, np.max(key_item), np.min(key_item))
    return mnist_content


def compress_archive(content_to_save, archive_name='./env/mnist.pkl'):
    save_file = open(archive_name, 'wb')
    pickle.dump(content_to_save, save_file)
    save_file.close()
    print_mnist_info('Archive saved to: ' + archive_name)
    

if __name__ == "__main__":
    download_mnist()
    mnist_content = load_mnist()
    compress_archive(mnist_content)



print('\033[91mFINISH: prepare_mnist\033[0m')



