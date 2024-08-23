import torch
from torch import optim
import torch.nn.functional as F
import argparse
import csv
import os
import numpy as np
from copy import deepcopy
from argparse import ArgumentParser

from models.network import Reg_Domain
import models.TransformerVSFA as TransformerVSFA
import logging
import random
import time

import h5py
import torch
import torch.nn as nn
from scipy import stats

import os.path
import pandas as pd
from models.network import Reg_Domain

from Dataload import VQADataset, VQADataset_test
from torch.utils.data import Dataset



class Logger:
    def __init__(self,dataset_str,exp_id,mode='w'):
         # 第一步，创建一个logger
        self.logger = logging.getLogger('log')
        self.logger.setLevel(logging.DEBUG)  # Log等级总开关
        # 第二步，创建一个handler，用于写入日志文件
        rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        log_path = '/home/d310/10t/wl/VQA_Model/Backbone/222222/44444/logs/'
        log_name = log_path + dataset_str+'_'+str(exp_id)+'_'+rq + '.log'
        logfile = log_name
        fh = logging.FileHandler(logfile, mode=mode)
        fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
        # 第三步，定义handler的输出格式
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        # 第四步，将logger添加到handler里面
        self.logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)  # 输出到console的log等级的开关
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

def randperm(length, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.permutation(length)



if __name__ == "__main__":
    parser = ArgumentParser(description='"VSFA: Quality Assessment of In-the-Wild Videos')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--lr', type=float, default=0.00005,
                        help='learning rate (default: 0.00001)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 2000)')

    parser.add_argument('--database', default='LIVE-VQC', type=str,
                        help='database name (default: CVD2014)')
    parser.add_argument('--model', default='/media/data/wl/VQA_Model/VSFA/models/VSFA.pt', type=str,
                        help='model name (default: VSFA)')
    parser.add_argument('--exp_id', default=0, type=int,
                        help='exp id for train-val-test splits (default: 0)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='test ratio (default: 0.2)')
    parser.add_argument("--notest_during_training", action='store_true',
                        help='flag whether to test during training')
    # parser.add_argument("--disable_visualization", action='store_true',
    #                     help='flag whether to enable TensorBoard visualization')
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    args = parser.parse_args()
    
    logger=Logger(args.database,args.exp_id)


    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True


    print('EXP ID: {}'.format(args.exp_id))
    print(args.database)
    # print(args.model)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    if args.database == 'LIVE-VQC':
        features_dir = '/home/d310/10t/wl/VQA_Model/Backbone/tt/'   # ECO

    pd_reader = pd.read_csv(datainfo)
    # print(pd_reader.get('video_name'))
    video_names = pd_reader.get('video_name')
    scores = pd_reader.get('MOS')


    ref_ids = [i for i in range(len(video_names))]
    max_len = 16
    scores = np.array(scores)
    scale = scores.max()
    index = randperm(len(scores))


    testindex = index[:]
    train_index, val_index, test_index = [], [], []

   

    for i in range(len(ref_ids)):
            test_index.append(video_names[i]) 

   
    test_dataset = VQADataset_test('/home/d310/10t/wl/VQA_Model/Backbone/tt/', test_index, max_len, scale=scale)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset)


    net =Reg_Domain(256,16, False)
    model = net.to(device)
    
    MSE_loss = torch.nn.L1Loss()
    
    
    model.load_state_dict(torch.load('/home/d310/10t/wl/VQA_Model/Backbone/Result/VQC/model/LIVE-VQC-EXP0.pt'))  #
    model.eval()
    with torch.no_grad():
        y_pred = np.zeros(len(test_index))
        y_test = np.zeros(len(test_index))
        names = []
        L = 0
        for i, (features, length, label,name) in enumerate(test_loader):
            names .append( name)
            y_test[i] = scale * label.item()  #
            features = features.to(device).float()
            label = label.to(device).float()
            outputs = model(features,  length)
            outputs = outputs.to(device)
            y_pred[i] = scale * outputs.item()
            loss = MSE_loss(outputs, label)
            L = L + loss.item()
    test_loss = L / (i + 1)

    print(names)
    print('*******')
    print(y_pred)

