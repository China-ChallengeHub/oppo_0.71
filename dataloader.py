import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import math
from tqdm import tqdm
from copy import copy, deepcopy


# dataLoader
class DatasetFolder(Dataset):

    def __init__(self, matData, data_an=True):

        if data_an:
            matData_an = np.concatenate([matData, 
                                     matData*-1,
                                    np.concatenate([matData[:,:,:,:,1:2]*-1, matData[:,:,:,:,0:1]], axis=-1),
                                    np.concatenate([matData[:,:,:,:,1:2], matData[:,:,:,:,0:1]*-1], axis=-1)], axis=0)
            
            self.matdata = matData_an
        else:
            self.matdata = matData

    def __getitem__(self, index):
        data = self.matdata[index]
        return data
        
    def __len__(self):
        return self.matdata.shape[0]
