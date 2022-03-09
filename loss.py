import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import math


class SmiLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss().cuda()

    def forward(self, output,input, epoch=0):
        input = input.reshape((-1,4*32*32, 2))
        input_complex = torch.complex(input[:,:,0], input[:,:,1])
        input = F.normalize(input_complex, p=2, dim=1)
        output = output.reshape((-1,4*32*32, 2))
        output_complex = torch.complex(output[:,:,0], output[:,:,1])
        output = F.normalize(output_complex, p=2, dim=1)
        sim = torch.abs(torch.sum(input * torch.conj(output), dim=1))
        loss = []
        sim_loss = 1.0 - torch.mean(sim * sim)
        
        loss.append(sim_loss)
        loss.append(self.mse(input.real, output.real))
        loss.append(self.mse(input.imag, output.imag))

        return loss
