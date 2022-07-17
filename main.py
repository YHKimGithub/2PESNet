import os
import json
import torch
import torchvision
import torch.nn.parallel
import torch.optim as optim
import numpy as np
import opts_thumos as opts
import time
import h5py
from eval import evaluation_detection
from module_mhd import *
from module_edr import *
from module_asd import *

def main(opt):
    if opt['module']=='mhd':
        if opt['mode'] == 'train':
            train_MHD(opt)
        if opt['mode'] == 'test':
            test_MHD(opt)
        if opt['mode'] == 'data_gen':
            generate_MHD_dataset(opt)
    if opt['module']=='edr':
        if opt['mode'] == 'train':
            train_EDR(opt)
        if opt['mode'] == 'test':
            test_EDR(opt)
    if opt['module']=='asd':
        if opt['mode'] == 'train':
            train_ASD(opt)
        if opt['mode'] == 'test':
            test_ASD(opt)
        if opt['mode'] == 'eval':
            evaluation_detection(opt)

if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"]) 
    opt_file=open(opt["checkpoint_path"]+"/opts.json","w")
    json.dump(opt,opt_file)
    opt_file.close()
    main(opt)
