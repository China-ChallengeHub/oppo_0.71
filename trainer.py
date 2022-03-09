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
import torch.nn.functional as F
from torch import autograd
from transformer import  *
from dataloader import *
from loss import *
import os
import torch.nn as nn
import pickle
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
epochs = 3000
learning_rate = 1e-3
num_workers = 1
print_freq = 250  # print frequency (default: 60)
# parameters for data
feedback_bits = 52  # sim:0.3541180491447449, multi:1.1020112783975629, multi_div_sim_1:3.111988448651818   sim:0.35078614950180054, multi:1.0383430611436908, multi_div_sim_1:2.9600457789407706
size_packet = 500

NUM_RX = 4
NUM_TX = 32
NUM_DELAY = 32
NUM_SAMPLE_TRAIN = 500



# Model construction
model = AutoEncoder(feedback_bits=52, dropout=0.2)

# model.encoder.load_state_dict(torch.load('submit_pt/encoder_1.pth.tar')['state_dict'])
# model.decoder.load_state_dict(torch.load('submit_pt/generator_1.pth.tar')['state_dict'])

if use_single_gpu:
    model = model.cuda()
else:
    # DataParallel will divide and allocate batch_size to all available GPUs
    autoencoder = torch.nn.DataParallel(model).cuda()

import scipy.io as scio
# criterion = FeaLoss()
criterion = SmiLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#optimizer = AdamWGC(model.parameters(), lr=learning_rate)
"""
scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer,
                                            T_max=epochs * len(train_loader),
                                            T_warmup=epochs//20 * len(train_loader),
                                            eta_min=1e-6)
"""

data_train = h5py.File('data/H1_32T4R.mat', 'r')
data_train = np.transpose(data_train['H1_32T4R'][:])
data_train = data_train[:, :, :, :, np.newaxis]
data_train = np.concatenate([data_train['real'], data_train['imag']], 4) # 500 4 32 32 2
# data_train = np.reshape(data_train, [NUM_SAMPLE_TRAIN, NUM_RX* NUM_TX, NUM_DELAY* 2, 1])
# x_train = norm_data(data_train, NUM_SAMPLE_TRAIN, NUM_RX, NUM_TX, NUM_DELAY)
data_train = data_train.astype(np.float32)
x_train = data_train

# Data loading

#x_train = np.concatenate((x_train,soft_test), axis=0)
"""
x_test = x_test[1000:,:,:,:]
"""

# dataLoader for training
train_dataset = DatasetFolder(x_train, data_an=True)
print(train_dataset.__len__())
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

best_loss = 100000
"""
scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer,
                                    T_max=epochs * len(train_loader),
                                    T_warmup=epochs//20 * len(train_loader),
                                    eta_min=1e-6)
"""
print('----', len(train_loader))


for epoch in range(epochs):

    if epoch == 300:
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
    # if epoch%10==0:
    if epoch >= 0:
        with torch.no_grad():
            for idx in range(int(500 / size_packet)):
                x = np.random.randint(2, size=(size_packet,feedback_bits))
                x = torch.from_numpy(x).float()
                x = x.cuda()
                input = x
                output = model.decoder(input)
                output = output.detach().cpu().numpy()
                if idx == 0:
                    output_all = output
                else:
                    output_all = np.concatenate([output_all, output], axis=0)

        new_output_all = output_all 

        real = data_train[:,:,:,:,0] + data_train[:,:,:,:,1]*1j
        fake = new_output_all[:,:,:,:,0] + new_output_all[:,:,:,:,1]*1j

        sim_1, multi_1, multi_div_sim_1 = K_nearest(real, fake, NUM_RX, NUM_TX, NUM_DELAY, 2)
        print('sim:{}, multi:{}, multi_div_sim_1:{}'.format(sim_1, multi_1, multi_div_sim_1))

        if multi_div_sim_1 < best_loss:
            modelSave2 = './submit_pt/generator_1.pth.tar'
            torch.save({'state_dict': model.decoder.state_dict(), }, modelSave2)
            modelSave2 = './submit_pt/encoder_1.pth.tar'
            torch.save({'state_dict': model.encoder.state_dict(), }, modelSave2)
            print("Model saved")
            best_loss = multi_div_sim_1
            
