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

from Dataload import VQADataset
from torch.utils.data import Dataset



class Logger:
    def __init__(self,dataset_str,exp_id,mode='w'):
         # 第一步，创建一个logger
        self.logger = logging.getLogger('log')
        self.logger.setLevel(logging.DEBUG)  # Log等级总开关
        # 第二步，创建一个handler，用于写入日志文件
        rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        log_path = '/home/d310/10t/wl/VQA_Model/Backbone/test/result/2_32_TSN/'
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
    parser.add_argument('--exp_id', default=200, type=int,
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

    if args.database == 'KoNViD-1k':
        # features_dir = '/home/d310/10t/wl/VQA_Model/Backbone/features/FasterNet_KoNViD-1k/'  # features dir
        datainfo = '/home/d310/10t/wl/VQA_Model/fuckvqa/data/KoNViD_1k/KoNViD_1k.csv'  # database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.
    if args.database == 'CVD2014':
        features_dir = '/home/d310/10t/wl/VQA_Model/Backbone/features/MobileNet_CVD2014/'
        datainfo = '/home/d310/10t/wl/VQA_Model/VSFA/data/CVD2014info.mat'
    if args.database == 'LIVE-Qualcomm':
        features_dir = '/home/d310/10t/wl/VQA_Model/Backbone/features/MobileNet_LIVE_Q/'
        datainfo = '/home/d310/10t/wl/Videos_Data/LIVE-Qualcomm/LIVEQualcomm.csv'                
    if args.database == 'LSVQ':
        features_dir = '/home/d310/10t/wl/VQA_Model/Backbone/features/MobileNet_LSVQ/'
        datainfo = '/home/d310/10t/wl/VQA_Model/VSFA/data/labels_train_test.csv' 
    if args.database == 'LIVE-VQC':
    #     # features_dir = '/home/d310/10t/wl/VQA_Model/Backbone/features/MobileNet_VQC/'   # TSN
        datainfo = '/home/d310/10t/wl/VQA_Model/VSFA/data/VQC1.csv'
    if args.database == 'NTAR':
        # features_dir = '/home/d310/10t/wl/VQA_Model/Backbone/features/MobileNet_NTAR/'
        datainfo = '/home/d310/10t/wl/Videos_Data/NTRA/train.txt' # labels_test_1080p.csv

    print('EXP ID: {}'.format(args.exp_id))
    print(args.database)
    # print(args.model)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    

    pd_reader = pd.read_csv(datainfo)
    # print(pd_reader.get('video_name'))
    video_names = pd_reader.get('video_name')
    scores = pd_reader.get('MOS')



    ref_ids = [i for i in range(len(video_names))]
    max_len = 70
    scores = np.array(scores)
    scale = scores.max()
    index = randperm(len(scores))

    trainindex = index[0:int(np.ceil((1 - args.test_ratio) * len(index)))]
    testindex = index[int(np.ceil((1 - args.test_ratio) * len(index))):len(index)]
    train_index, val_index, test_index = [], [], []


    for i in range(len(ref_ids)):
        if (ref_ids[i] in trainindex):
            train_index.append(video_names[i])  
        else:
            test_index.append(video_names[i]) 

    train_dataset = VQADataset('/home/d310/10t/wl/VQA_Model/Backbone/test/2_32_TSN/', train_index, max_len, scale=scale)
 
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    if args.test_ratio > 0:
        test_dataset = VQADataset('/home/d310/10t/wl/VQA_Model/Backbone/test/2_32_TSN/', test_index, max_len, scale=scale)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset)

    # /home/d310/10t/wl/VQA_Model/1/TSN  顺序


    if not os.path.exists('/home/d310/10t/wl/VQA_Model/Backbone/test/result/2_32_TSN'):
        os.makedirs('/home/d310/10t/wl/VQA_Model/Backbone/test/result/2_32_TSN')
    trained_model_file = '/home/d310/10t/wl/VQA_Model/Backbone/test/result/2_32_TSN/{}-EXP{}.pt'.format(args.database, args.exp_id)
    if not os.path.exists('/home/d310/10t/wl/VQA_Model/Backbone/test/result/2_32_TSN'):
        os.makedirs('/home/d310/10t/wl/VQA_Model/Backbone/test/result/2_32_TSN')
    save_result_file = '/home/d310/10t/wl/VQA_Model/Backbone/test/result/2_32_TSN/{}-EXP{}'.format(args.database, args.exp_id)


    net =Reg_Domain(320,16, False)

    model = net.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=0.000001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    MSE_loss = torch.nn.L1Loss()
    best_test_criterion = -1
    logger.logger.info('start training......\n')
    for epoch in range(args.epochs):
        # Train
        model.train()
        L = 0
        for i, (features, length, label) in enumerate(train_loader):
            features = features.to(device).float()
            label = label.to(device).float()
            optimizer.zero_grad()  #
            outputs = model(features, length)
            outputs = outputs.to(device)
            loss = MSE_loss(outputs, label.squeeze())
            loss.backward()
            optimizer.step()
            scheduler.step()
            L = L + loss.item()

        train_loss = L / (i + 1)

        model.eval()

        # Test
        if args.test_ratio > 0 and not args.notest_during_training:
            y_pred = np.zeros(len(test_index))
            y_test = np.zeros(len(test_index))
            L = 0
            with torch.no_grad():
                for i, (features, length, label) in enumerate(test_loader):
                    y_test[i] = scale * label.item()  #
                    features = features.to(device).float()
                    label = label.to(device).float()
                    outputs = model(features,  length)
                    outputs = outputs.to(device)
                    y_pred[i] = scale * outputs.item()
                    loss = MSE_loss(outputs, label)
                    L = L + loss.item()
            test_loss = L / (i + 1)
        PLCC = stats.pearsonr(y_pred, y_test)[0]
        SROCC = stats.spearmanr(y_pred, y_test)[0]
        RMSE = np.sqrt(((y_pred-y_test) ** 2).mean())
        KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
        
        
        logger.logger.info('%d epoch train loss: %.4f \n'%(epoch,train_loss))
        logger.logger.info('%d epoch test loss: %.4f \n'%(epoch,test_loss))
        logger.logger.info('%d epoch test SRCC: %.2f \n'%(epoch,SROCC))
        logger.logger.info('%d epoch test KRCC: %.2f \n'%(epoch,KROCC))
        logger.logger.info('%d epoch test PLCC: %.2f \n'%(epoch,PLCC))
        logger.logger.info('%d epoch test RMSE: %.2f \n'%(epoch,RMSE))

        # Update the model with the best val_SROCC
        if SROCC > best_test_criterion:
            print("EXP ID={}: Update best model using best_val_criterion in epoch {}".format(args.exp_id, epoch))
    
            print("Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                      .format(test_loss, SROCC, KROCC, PLCC, RMSE))
                # np.save(save_result_file, (y_pred, y_test, test_loss, SROCC, KROCC, PLCC, RMSE, test_index))
            torch.save(model.state_dict(), trained_model_file)
            best_test_criterion = SROCC  # update best val SROCC
            logger.logger.info('best model in %d epoch,SRCC: %.2f\n'%(epoch,SROCC))
            logger.logger.info('best model in %d epoch,KRCC: %.2f\n'%(epoch,KROCC))
            logger.logger.info('best model in %d epoch,PLCC: %.2f\n'%(epoch,PLCC))
            logger.logger.info('best model in %d epoch,RMSE: %.2f\n'%(epoch,RMSE))
            logger.logger.info('save model %d successed......\n'%epoch)

    # Test
    if args.test_ratio > 0:
        model.load_state_dict(torch.load(trained_model_file))  #
        model.eval()
        with torch.no_grad():
            y_pred = np.zeros(len(test_index))
            y_test = np.zeros(len(test_index))
            L = 0
            for i, (features, length, label) in enumerate(test_loader):
                y_test[i] = scale * label.item()  #
                features = features.to(device).float()
                label = label.to(device).float()
                outputs = model(features,  length)
                outputs = outputs.to(device)
                y_pred[i] = scale * outputs.item()
                loss = MSE_loss(outputs, label)
                L = L + loss.item()
        test_loss = L / (i + 1)
        PLCC = stats.pearsonr(y_pred, y_test)[0]
        SROCC = stats.spearmanr(y_pred, y_test)[0]
        RMSE = np.sqrt(((y_pred-y_test) ** 2).mean())
        KROCC = stats.stats.kendalltau(y_pred, y_test)[0]

        print("Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
              .format(test_loss, SROCC, KROCC, PLCC, RMSE))
        np.save(save_result_file, (y_pred, y_test, test_loss, SROCC, KROCC, PLCC, RMSE, test_index))
        logger.logger.info('Finally test result:SRCC: %.2f\n'%(SROCC))
        logger.logger.info('Finally test result:KRCC: %.2f\n'%(KROCC))
        logger.logger.info('Finally test result:PLCC: %.2f\n'%(PLCC))
        logger.logger.info('Finally test result:RMSE: %.2f\n'%(RMSE))

