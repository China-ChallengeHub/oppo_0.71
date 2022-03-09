#!/usr/bin/env python3
"""An Implement of an autoencoder with pytorch.
This is the template code for 2020 NIAC https://naic.pcl.ac.cn/.
The code is based on the sample code with tensorflow for 2020 NIAC and it can only run with GPUS.
Note:
    1.This file is used for designing the structure of encoder and decoder.
    2.The neural network structure in this model file is CsiNet, more details about CsiNet can be found in [1].
[1] C. Wen, W. Shih and S. Jin, "Deep Learning for Massive MIMO CSI Feedback", in IEEE Wireless Communications Letters, vol. 7, no. 5, pp. 748-751, Oct. 2018, doi: 10.1109/LWC.2018.2818160.
    3.The output of the encoder must be the bitstream.
"""
import numpy as np
import h5py
import torch
from transformer import  *
from dataloader import *
from loss import *
import os
import torch.nn as nn
import pickle
import random
from copy import deepcopy
from util import *
def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(  name + '.pkl', 'rb') as f:
        return pickle.load(f)
# Parameters for training
torch.backends.cudnn.benchmark=True
np.set_printoptions(suppress=True)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
use_single_gpu = True  # select whether using single gpu or multiple gpus
torch.manual_seed(1)
batch_size = 16
epochs = 8000
learning_rate = 1e-3
num_workers = 4
print_freq = 3200  # print frequency (default: 60)
# parameters for data
# feedback_bits = 48  # sim2 = 0.26814688715868884, multi2 = 2.3217122101795313, multi_div_sim_2 = 8.658359732535498
feedback_bits = 48
B = 2
size_packet = 100

NUM_RX = 4
NUM_TX = 32
NUM_DELAY = 32
NUM_SAMPLE_TRAIN = 4000



def norm_data(x, num_sample, num_rx, num_tx, num_delay):
    x2 = np.reshape(x, [num_sample, num_rx * num_tx * num_delay * 2])
    x_max = np.max(abs(x2), axis=1)
    x_max = x_max[:,np.newaxis]
    x3 = x2 / x_max / 2.0
    y = np.reshape(x3, [num_sample, num_rx, num_tx, num_delay, 2])
    return y


# Model construction
model = AutoEncoder(feedback_bits=48, dropout=0.1)

# model.encoder.load_state_dict(torch.load('submit_pt/encoder_2.pth.tar')['state_dict'])
# model.decoder.load_state_dict(torch.load('submit_pt/generator_2.pth.tar')['state_dict'])


if use_single_gpu:
    model = model.cuda()
else:
    # DataParallel will divide and allocate batch_size to all available GPUs
    model = torch.nn.DataParallel(model).cuda()

import scipy.io as scio
criterion = SmiLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
"""
scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer,
                                            T_max=epochs * len(train_loader),
                                            T_warmup=epochs//20 * len(train_loader),
                                            eta_min=1e-6)
"""

data_train = h5py.File('data/H2_32T4R.mat', 'r')
data_train = np.transpose(data_train['H2_32T4R'][:])
data_train = data_train[:, :, :, :, np.newaxis]
data_train = np.concatenate([data_train['real'], data_train['imag']], 4) # 500 4 32 32 2
# data_train = np.reshape(data_train, [NUM_SAMPLE_TRAIN, NUM_RX* NUM_TX, NUM_DELAY* 2, 1])
# x_train = norm_data(data_train, NUM_SAMPLE_TRAIN, NUM_RX, NUM_TX, NUM_DELAY)
data_train = data_train.astype(np.float32)

x_train = data_train 
x_test = data_train


"""
x_test = x_test[1000:,:,:,:]
"""


x_train_hat = np.transpose(x_train, (0,3,1,2,4)).reshape(-1, 32, 256)
x_train_paterns = np.unique((np.sum(np.abs(x_train_hat), axis=2) != 0).astype(np.float32), axis=0)

# dataLoader for training
train_dataset = DatasetFolder(x_train, data_an=True)
print(train_dataset.__len__())
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)


def random_cutmix(x):
    a = x[:,:,:,:16,:]
    b = x[:,:,:,16:,:]
    idx = [i for i in range(a.size(0))]
    random.shuffle(idx)
    res = torch.cat([a,b[idx]], dim=3)
    return res


def random_mixup(x, num=3):
    weight = np.random.randn(num)
    weight = weight / np.sum(weight)
    res = x * weight[0]
    for i in range(1, num):
        idx = [i for i in range(x.size(0))]
        random.shuffle(idx)
        res += x[idx] * weight[i]
    return res


best_loss = 100000
"""
scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer,
                                    T_max=epochs * len(train_loader),
                                    T_warmup=epochs//20 * len(train_loader),
                                    eta_min=1e-6)
"""
print('----', len(train_loader))
for epoch in range(epochs):

    if epoch == 200:
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate * 0.1

    # model training
    model.train()
    total_loss = []
    for i, x in enumerate(train_loader):
        input = x
        input = input.cuda()
        # compute output
        output = model(input)

        loss_list = criterion(output, input, epoch=epoch)
        loss = sum(loss_list[1:])
        total_loss.append([item.detach().cpu().numpy()*input.size(0) for item in loss_list])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.6f}\t'.format(
                epoch, i, len(train_loader), loss=loss.item()))
    # model evaluating
    total_loss = np.sum(np.array(total_loss), axis=0) / len(train_dataset)
    print('train loss:{},  other loss:{}\t'.format(total_loss[1], total_loss[2]))
    model.eval()
    total_loss = []
    totalNMSE = 0
    y_test = []
    count = 0
    if epoch%1==0:
        with torch.no_grad():
            for idx in range(int(4000 / size_packet)):
                x = np.random.randint(2, size=(size_packet,feedback_bits))
                x = torch.from_numpy(x)
                x = x.cuda()
                """
                B,_,_,_ = x.size()
                x_var = torch.mean((x.view(B,126,-1).detach() - 0.5)**2,dim = -1)
                x_sort = torch.sort(-x_var,dim = -1)[1] + torch.arange(B).unsqueeze(-1).to(x_var.device)*126
                x_sort = x_sort.view(-1)
                x = x.view(B*126,128,2)
                input = torch.index_select(x, 0, x_sort).view(B,2,126,128)
                """
                input = x
                output = model.decoder(input)   # bx4x32x32x2

                output = output.detach().cpu().numpy()
                if idx == 0:
                    output_all = output
                else:
                    output_all = np.concatenate([output_all, output], axis=0)

        new_output_all = output_all

        real = x_test[:,:,:,:,0] + x_test[:,:,:,:,1]*1j
        fake = new_output_all[:,:,:,:,0] + new_output_all[:,:,:,:,1]*1j

        sim_1, multi_1, multi_div_sim_1 = K_nearest(real, fake, NUM_RX, NUM_TX, NUM_DELAY, 2)
        print('sim:{}, multi:{}, multi_div_sim_1:{}'.format(sim_1, multi_1, multi_div_sim_1))

        if multi_div_sim_1 < best_loss:
            modelSave2 = './submit_pt/generator_2.pth.tar'
            torch.save({'state_dict': model.decoder.state_dict(), }, modelSave2)
            modelSave2 = './submit_pt/encoder_2.pth.tar'
            torch.save({'state_dict': model.encoder.state_dict(), }, modelSave2)

            print("Model saved")
            best_loss = multi_div_sim_1
            
