# from models.fasternet import FasterNet
# from models.van import VAN
# import  models.MobileOne as mobileone
# import models.ShuffeNet as SH

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
import skvideo.io
from PIL import Image
import os
import h5py
import numpy as np
import random
from argparse import ArgumentParser
import pandas as pd
from Dataload import VideoDataset
import Sample
# from fvcore.nn import FlopCountAnalysis
def get_features(video_data, device='cuda'):
    """feature extraction"""

    extractor = models.mobilenet_v2(pretrained=True).features[:18].to(device)
  
    video_length = len(video_data)
    frame_start = 0
    output1 = torch.Tensor().to(device)
    extractor.eval()
    conv = nn.Conv2d(320, 2048, kernel_size=1).to(device) 
    liner = nn.Linear(256,320).to(device)
    
    with torch.no_grad(): 
        while frame_start < video_length:
            ten_num = Sample.get_spatial_fragments_5_5(video_data[frame_start])
            ten_num = ten_num.to(device) 
            if ten_num.shape[0] == 0:
                frame_start = frame_start + 1
            else:
                features = extractor(ten_num)
                output1 = torch.cat((output1, features), 0)
                frame_start = frame_start + 1


        out = output1
    return out



if __name__ == '__main__':
    
    parser = ArgumentParser(description='"Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--database', default='LIVE-VQC', type=str,
                        help='database name (default: KoNViD-1k)')
    parser.add_argument('--frame_batch_size', type=int, default=64,
                        help='frame batch size for feature extraction (default: 64)')

    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    args = parser.parse_args()

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    if args.database == 'KoNViD-1k':
        videos_dir = '/home/d310/10t/wl/Videos_Data/KoNViD_1k_videos/KoNViD_1k_videos/'  # videos dir
        features_dir = '/home/d310/10t/wl/VQA_Model/Backbone/222222/'  # features dir
        # datainfo = '/home/d310/10t/wl/VQA_Model/fuckvqa/data/KoNViD_1k/KoNViD_1k.csv'  # database info: v
        datainfo = '/home/d310/10t/wl/Videos_Data/KoNViD_1k_videos/1.csv'
    

    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    pd_reader = pd.read_csv(datainfo)
    # print(pd_reader.get('video_name'))
    video_names = pd_reader.get('video_name')
    scores = pd_reader.get('MOS')
    video_format ='RGB'
    width = None
    height = None



    dataset = VideoDataset(videos_dir, video_names, scores, video_format, width, height)
    for i in range(len(dataset)):
        current_data = dataset[i]       
        current_video = current_data['video']
        current_score = current_data['score']
        name = current_data['name']
        print('Video {}: length {}'.format(i, current_video.shape[0]))
        features =  get_features(current_video)
        np.save(features_dir + name + '_resnet-50_res5c', features.to('cpu').numpy())
        np.save(features_dir + name + '_score', current_score)
        # CUDA_VISIBLE_DEVICES=3 python GetFeatures.py --database=KoNViD-1k

