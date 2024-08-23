from argparse import ArgumentParser
import os
import h5py
import torch
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random
from scipy import stats
# import datetime
from config import  *
import logging  # 引入logging模块
import os.path
import time
import pandas as pd
import models


class VQADataset(Dataset):
    def __init__(self, features_dir, index=None, max_len=240, feat_dim=4096, scale=1):
        super(VQADataset, self).__init__()
        self.features = np.zeros((len(index), max_len, feat_dim))
        self.length = np.zeros((len(index), 1))
        self.mos = np.zeros((len(index), 1))
        for i in range(len(index)):
            features = np.load(features_dir + str(index[i]) + '_resnet-50_res5c.npy')   # [49,2049]
            self.length[i] = features.shape[0]
            self.features[i, :features.shape[0], :] = features
            self.mos[i] = np.load(features_dir + str(index[i]) + '_score.npy')  #
        self.scale = scale  #
        self.label = self.mos / self.scale  # label normalization

    def __len__(self):
        return len(self.mos)

    def __getitem__(self, idx):
        sample = self.features[idx], self.length[idx], self.label[idx]
        return sample


class Logger:
    def __init__(self,dataset_str,exp_id,mode='w'):
         # 第一步，创建一个logger
        self.logger = logging.getLogger('log')
        self.logger.setLevel(logging.DEBUG)  # Log等级总开关
        # 第二步，创建一个handler，用于写入日志文件
        rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        log_path = os.getcwd() + '/Results_Data/6*6_TSM/logs/'
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


if __name__ == "__main__":
    parser = ArgumentParser(description='"VSFA: Quality Assessment of In-the-Wild Videos')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--lr', type=float, default=0.00001,
                        help='learning rate (default: 0.00001)')
    parser.add_argument('--batch_size', type=int, default=16,
                        
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='number of epochs to train (default: 2000)')

    parser.add_argument('--database', default='LIVE-Qualcomm', type=str,
                        help='database name (default: CVD2014)')
    parser.add_argument('--model', default='/media/data/wl/VQA_Model/VSFA/models/VSFA.pt', type=str,
                        help='model name (default: VSFA)')
    parser.add_argument('--exp_id', default=3, type=int,
                        help='exp id for train-val-test splits (default: 0)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='test ratio (default: 0.2)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='val ratio (default: 0.2)')

    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')

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

    args.decay_interval = int(args.epochs/10)
    args.decay_ratio = 0.8

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    if args.database == 'KoNViD-1k':
        features_dir = '/media/data/wl/CNNfeatures/6*6_TSM_1/CNN_feature_KoNViD-1k/'  # features dir
        datainfo = '/media/data/wl/VQA_Model/VSFA/data/KoNViD-1kinfo.mat'  # database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.
    if args.database == 'CVD2014':
        features_dir = '/media/data/wl/CNNfeatures/6*6_TSM_1/CVD2014/'
        datainfo = '/media/data/wl/VQA_Model/VSFA/data/CVD2014info.mat'
    if args.database == 'LIVE-Qualcomm':
        features_dir = '/home/test/10t/wl/CNNfeatures/6*6_TSM_1/LIVE-Qualconmm/'
        datainfo = '/media/data/wl/VQA_Model/VSFA/data/LIVE-Qualcomminfo.mat'


    print('EXP ID: {}'.format(args.exp_id))
    print(args.database)
    print(args.model)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    Info = h5py.File(datainfo, 'r')  # index, ref_ids
    index = Info['index']
    index = index[:, args.exp_id % index.shape[1]]  # np.random.permutation(N)
    ref_ids = Info['ref_ids'][0, :]  #
    max_len = int(Info['max_len'][0])
    trainindex = index[0:int(np.ceil((1 - args.test_ratio - args.val_ratio) * len(index)))]
    testindex = index[int(np.ceil((1 - args.test_ratio) * len(index))):len(index)]
    train_index, val_index, test_index = [], [], []
    for i in range(len(ref_ids)):
        train_index.append(i) if (ref_ids[i] in trainindex) else \
            test_index.append(i) if (ref_ids[i] in testindex) else \
                val_index.append(i)

    scale = Info['scores'][0, :].max()  # label normalization factor

    trainindex = index[0:int(np.ceil((1 - args.test_ratio - args.val_ratio) * len(index)))]
    testindex = index[int(np.ceil((1 - args.test_ratio) * len(index))):len(index)]
    train_index, val_index, test_index = [], [], []
    for i in range(len(ref_ids)):
        train_index.append(i) if (ref_ids[i] in trainindex) else \
            test_index.append(i) if (ref_ids[i] in testindex) else \
                val_index.append(i)

    train_dataset = VQADataset(features_dir, train_index, max_len, scale=scale)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = VQADataset(features_dir, val_index, max_len, scale=scale)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset)
    if args.test_ratio > 0:
        test_dataset = VQADataset(features_dir, test_index, max_len, scale=scale)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset)
                
    model = getattr(models, 'TransformerVSFA')(input_size=4096, max_len=max_len, n_layers=3, n_heads=2, d_k=64, d_v=64).to(device)
    model = torch.nn.DataParallel(model)
    # print(model)
    # hyper params

    if not os.path.exists('/media/data/wl/VQA_Model/fuckvqa/Results_Data/6*6_TSM/model'):
        os.makedirs('/media/data/wl/VQA_Model/fuckvqa/Results_Data/6*6_TSM/model')
    trained_model_file = '/media/data/wl/VQA_Model/fuckvqa/Results_Data/6*6_TSM/model/{}-EXP{}.pt'.format(args.database, args.exp_id)
    if not os.path.exists('/media/data/wl/VQA_Model/fuckvqa/Results_Data/6*6_TSM/results'):
        os.makedirs('/media/data/wl/VQA_Model/fuckvqa/Results_Data/6*6_TSM/results')
    save_result_file = '/media/data/wl/VQA_Model/fuckvqa/Results_Data/6*6_TSM/results/{}-EXP{}'.format(args.database, args.exp_id)



    criterion = nn.L1Loss()  # L1 loss
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_criterion = -1  # SROCC min
    logger.logger.info('start training......\n')
    for epoch in range(args.epochs):
        # Train
        model.train()
        L = 0
        for i, (features, length, label) in enumerate(train_loader):
            features = features.to(device).float()
            label = label.to(device).float()
            optimizer.zero_grad()  #
            outputs, attn= model(features, length.float())
            # print(type(outputs))
            # print(outputs.shape)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            L = L + loss.item()

        train_loss = L / (i + 1)

        model.eval()
        # Val
        y_pred = np.zeros(len(val_index))
        y_val = np.zeros(len(val_index))
        L = 0
        with torch.no_grad():
            for i, (features, length, label) in enumerate(val_loader):
                y_val[i] = scale * label.item()  #
                features = features.to(device).float()
                label = label.to(device).float()
                outputs, attn = model(features, length.float())
                y_pred[i] = scale * outputs.item()
                loss = criterion(outputs, label)
                L = L + loss.item()
        val_loss = L / (i + 1)
        val_PLCC = stats.pearsonr(y_pred, y_val)[0]
        val_SROCC = stats.spearmanr(y_pred, y_val)[0]
        val_RMSE = np.sqrt(((y_pred-y_val) ** 2).mean())
        val_KROCC = stats.stats.kendalltau(y_pred, y_val)[0]

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
                    outputs,attn = model(features, length.float())
                    y_pred[i] = scale * outputs.item()
                    loss = criterion(outputs, label)
                    L = L + loss.item()
            test_loss = L / (i + 1)
            PLCC = stats.pearsonr(y_pred, y_test)[0]
            SROCC = stats.spearmanr(y_pred, y_test)[0]
            RMSE = np.sqrt(((y_pred-y_test) ** 2).mean())
            KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
        
        
        logger.logger.info('%d epoch train loss: %.4f \n'%(epoch,train_loss))
        logger.logger.info('%d epoch Val loss: %.4f, SRCC: %.4f, KRCC:%.4f, PLCC: %.4f, RMSE:%.4f . \n'%
                           (epoch,val_loss, val_SROCC,val_KROCC, val_PLCC, val_RMSE))

        # Update the model with the best val_SROCC
        if val_SROCC > best_val_criterion:
            print("EXP ID={}: Update best model using best_val_criterion in epoch {}".format(args.exp_id, epoch))
            print("Val results: val loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                  .format(val_loss, val_SROCC, val_KROCC, val_PLCC, val_RMSE))
            if args.test_ratio > 0 and not args.notest_during_training:
                print("Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                      .format(test_loss, SROCC, KROCC, PLCC, RMSE))
                # np.save(save_result_file, (y_pred, y_test, test_loss, SROCC, KROCC, PLCC, RMSE, test_index))
            torch.save(model.state_dict(), trained_model_file)
            best_val_criterion = val_SROCC  # update best val SROCC
            logger.logger.info('best model in %d epoch,SRCC: %.4f,  KRCC:%.4f, PLCC: %.4f, RMSE:%.4f . \n'%
                           (epoch, SROCC,KROCC, PLCC, RMSE))
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
                outputs,attn = model(features, length.float())
                y_pred[i] = scale * outputs.item()
                loss = criterion(outputs, label)
                L = L + loss.item()
        test_loss = L / (i + 1)
        PLCC = stats.pearsonr(y_pred, y_test)[0]
        SROCC = stats.spearmanr(y_pred, y_test)[0]
        RMSE = np.sqrt(((y_pred-y_test) ** 2).mean())
        KROCC = stats.stats.kendalltau(y_pred, y_test)[0]

        print("Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
              .format(test_loss, SROCC, KROCC, PLCC, RMSE))
        np.save(save_result_file, (y_pred, y_test, test_loss, SROCC, KROCC, PLCC, RMSE, test_index))
        logger.logger.info('Finally test result:PLCC: %.4f, SRCC: %.4f KRCC:%.4f,  RMSE:%.4f . \n'%
                           (PLCC,SROCC,KROCC, RMSE))
