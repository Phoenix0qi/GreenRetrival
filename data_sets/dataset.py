import numpy as np
import glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
import os
import gzip
import h5py
from scipy.signal import savgol_filter

class Indoor_5G_scatters(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file = root_dir

        self.data_all = self._load_data(self.file) # tloc, rloc, gtot, scat

        self.data = self.data_all[0:2]

        self.label = self.data_all[2:4]



    def __len__(self):
        return len(self.data[0])

    def _load_data(self, data_file):
        f = h5py.File(data_file, 'r')
        tloc = f['dataset']['tloc'][...]
        rloc = f['dataset']['rloc'][...]
        # gtot = f['dataset']['gtot'][...] / 1000.0 # cube: /1000  indoor:/0.01
        gtot = f['dataset']['gtot'][...] / 0.01
        scat = f['dataset']['scat'][...]

        f.close()
        return tloc.astype(np.float32), rloc.astype(np.float32), gtot.astype(np.float32), scat.astype(np.float32)  # batch, 3 # batch, 3 # batch, 2 # batch, 10, 9


    def __getitem__(self, idx):
        data = [self.data[0][idx], self.data[1][idx]] # tloc, rloc
        label = [self.label[0][idx], self.label[1][idx]]  # gtot, scat

        return data, label

class pazhou_34e8_scatters(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file = root_dir

        self.data_all = self._load_data(self.file) # tloc, rloc, gtot, scat

        self.data = self.data_all[0:2]

        self.label = self.data_all[2:4]


    def __len__(self):
        return len(self.data[0])

    def _load_data(self, data_file):
        f = h5py.File(data_file, 'r')
        tloc = f['dataset']['tloc'][...]
        rloc = f['dataset']['rloc'][...]
        # gtot = f['dataset']['gtot'][...] / 1000.0 # cube: /1000  indoor:/0.01  bar: *100     sionna_PKU  model/100 gtot*1000  *3e-5
        gtot = f['dataset']['gtot'][...] / 1000 # * 1000 # * 10000 sionna # indoor 3G 3cm / 1000 # sionna_PKU * 1000  # pazhou * 100
        # scat = f['dataset']['scat'][...]

        # gtot = (f['dataset']['gtot'][...] - f['dataset']['scat'][...] )* 100

        scat = [[-1, -1, -1]] * len(tloc)

        f.close()
        return tloc.astype(np.float32), rloc.astype(np.float32), gtot.astype(np.float32), np.array(scat)  # batch, 3 # batch, 3 # batch, 2 # batch, 10, 9


    def __getitem__(self, idx):
        data = [self.data[0][idx], self.data[1][idx]] # tloc, rloc
        label = [self.label[0][idx], self.label[1][idx]]  # gtot, scat

        return data, label

class cube_3G_sameas_NeRF2(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file = root_dir

        self.data_all = self._load_data(self.file) # tloc, rloc, gtot, scat

        self.data = self.data_all[0:2]

        self.label = self.data_all[2:4]



    def __len__(self):
        return len(self.data[0])

    def _load_data(self, data_file):
        f = h5py.File(data_file, 'r')
        tloc = f['dataset']['tloc'][...]
        rloc = f['dataset']['rloc'][...]
        gtot = f['dataset']['gtot'][...] / 1000.0 # cube: /1000  indoor:/0.01
        # gtot = f['dataset']['gtot'][...] / 0.01
        scat = [[-1, -1, -1]] * len(tloc)

        f.close()
        return tloc.astype(np.float32), rloc.astype(np.float32), gtot.astype(np.float32), np.array(scat)


    def __getitem__(self, idx):
        data = [self.data[0][idx], self.data[1][idx]] # tloc, rloc
        label = [self.label[0][idx], self.label[1][idx]]  # gtot, scat

        return data, label

class Cube_3G_e5_sample_fine(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file = root_dir

        self.data_all = self._load_data(self.file) # tloc, rloc, gtot, scat

        self.data = self.data_all[0:2]

        self.label = self.data_all[2:4]



    def __len__(self):
        return len(self.data[0])

    def _load_data(self, data_file):
        f = h5py.File(data_file, 'r')  # 取不同比例 0.5 0.25   0.x * len(tloc)
        lens = len(f['dataset']['tloc'][...])
        tloc = f['dataset']['tloc'][0: int(lens*1)] # *1 *0.5
        rloc = f['dataset']['rloc'][0: int(lens*1)]
        gtot = f['dataset']['gtot'][0: int(lens*1)] / 1000.0 # cube: /1000  indoor:/0.01
        # gtot = f['dataset']['gtot'][...] / 0.01
        scat = [[-1, -1, -1]] * len(tloc)

        f.close()
        return tloc.astype(np.float32), rloc.astype(np.float32), gtot.astype(np.float32), np.array(scat)


    def __getitem__(self, idx):
        data = [self.data[0][idx], self.data[1][idx]] # tloc, rloc
        label = [self.label[0][idx], self.label[1][idx]]  # gtot, scat

        return data, label

class Select_Dataset():
    def __init__(self, data_type):
        self.data_type = data_type
        if self.data_type == 'Indoor_5G_scatters':
            self.dataset = Indoor_5G_scatters
        elif self.data_type == 'pazhou_34e8_scatters':
            self.dataset = pazhou_34e8_scatters
        elif self.data_type == 'cube_3G_sameas_NeRF2':
            self.dataset = cube_3G_sameas_NeRF2
        elif self.data_type == 'Cube_3G_e5_sample_fine':
            self.dataset = Cube_3G_e5_sample_fine


    def __call__(self):
        return self.dataset