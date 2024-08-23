"""Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50"""
# Author: Dingquan Li
# Email: dingquanli AT pku DOT edu DOT cn
# Date: 2018/3/27
# 
# CUDA_VISIBLE_DEVICES=0 python CNNfeatures.py --database=KoNViD-1k --frame_batch_size=64
# CUDA_VISIBLE_DEVICES=1 python CNNfeatures.py --database=CVD2014 --frame_batch_size=32
# CUDA_VISIBLE_DEVICES=0 python CNNfeatures.py --database=LIVE-Qualcomm --frame_batch_size=8

# LIVE_VQC
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
import Temporl
import Spatial
import pandas as pd


class VideoDataset(Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, videos_dir, video_names, score, video_format='RGB', width=None, height=None):

        super(VideoDataset, self).__init__()
        self.videos_dir = videos_dir
        self.video_names = [video_name+'.mp4' for video_name in video_names]
        self.score = score
        self.format = video_format
        self.width = width
        self.height = height

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        assert self.format == 'YUV420' or self.format == 'RGB'
        if self.format == 'YUV420':
            video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name), self.height, self.width, inputdict={'-pix_fmt':'yuvj420p'})
        else:
            video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name))
        video_score = self.score[idx]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        video_length = video_data.shape[0]
        video_channel = video_data.shape[3]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]
        
        #  Temporal Sampling
        num = Temporl.ECO(video_data)
        
        transformed_video = torch.zeros([len(num), video_channel,  video_height, video_width])
        for i, frame_idx in enumerate(num):
            frame = video_data[frame_idx]

            frame = Image.fromarray(frame)
            frame = transform(frame)
            transformed_video[i] = frame

        sample = {'video': transformed_video,
                  'score': video_score}

        return sample


class ResNet50(torch.nn.Module):
    """Modified ResNet50 for feature extraction"""
    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        # features@: 7->res5c
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == 7:
                features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
                features_std = global_std_pool2d(x)
                return features_mean, features_std


def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


from fvcore.nn import FlopCountAnalysis

def get_features(video_data, device='cuda'):
    """feature extraction"""
    extractor = ResNet50().to(device)

    total_params = sum(p.numel() for p in extractor.parameters())
    flops = FlopCountAnalysis(extractor, Spatial.get_spatial_fragments_5_5(video_data[0]).to(device))

    total_flops = flops.total()
    print(f"Total parameters: {total_params}")
    print(f"Total FLOPs: {total_flops}")
    video_length = len(video_data)
    frame_start = 0
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    extractor.eval()

    with torch.no_grad(): 
        while frame_start < video_length:
            #  Spatial Sampling
            
            ten_num = Spatial.get_spatial_fragments_2_2(video_data[frame_start])
            
            ten_num = ten_num.to(device) 
            # print(ten_num.shape)  # [3,3,32,32]
            if ten_num.shape[0] == 0:
                frame_start = frame_start + 1
            else:
                features_mean, features_std = extractor(ten_num)
                output1 = torch.cat((output1, features_mean), 0)
                output2 = torch.cat((output2, features_std), 0)
                frame_start = frame_start + 1

        output = torch.cat((output1, output2), 1).squeeze()

    return output

if __name__ == "__main__":
    parser = ArgumentParser(description='"Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--database', default='LSVQ1080p', type=str,
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

    if args.database == 'LIVE-VQC':
        videos_dir = '/media/data/wl/Videos_Data/LIVE-VQC/Video/'
        features_dir = '/media/data/wl/VQA_Model/VSFA/Lovc_6*6_c/'
        datainfo = '/media/data/wl/VQA_Model/VSFA/data/VQC.csv'
    if args.database == 'LSVQ':
        videos_dir = '/home/test/10t/wl/Videos_Data/LSVQ/'  # yfcc-batch
        features_dir = '/home/test/10t/wl/CNNfeatures/2*2_ECO_64/LSVQ/'
        datainfo = '/home/test/10t/wl/VQA_Model/VSFA/data/labels_train_test.csv' # labels_test_1080p.csv
    if args.database == 'LSVQ1080p':
        videos_dir = '/home/d310/10t/wl/Videos_Data/LSVQ/'  # yfcc-batch
        features_dir = '/home/d310/10t/wl/CNNfeatures/2*2_TSM_64/LSVQ_1080p/'
        datainfo = '/home/d310/10t/wl/VQA_Model/VSFA/data/labels_test_1080p.csv' # labels_test_1080p.csv

    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    pd_reader = pd.read_csv(datainfo)
    # print(pd_reader.get('video_name'))
    video_names = pd_reader.get('name')
    scores = pd_reader.get('mos')
    video_format ='RGB'
    width = None
    height = None
    dataset = VideoDataset(videos_dir, video_names, scores, video_format, width, height)

    for i in range(len(dataset)):
        current_data = dataset[i]       
        current_video = current_data['video']
        current_score = current_data['score']
        print('Video {}: length {}'.format(i, current_video.shape[0]))
        features = get_features(current_video, device) 
        np.save(features_dir + str(i) + '_resnet-50_res5c', features.to('cpu').numpy())
        np.save(features_dir + str(i) + '_score', current_score)
        # CUDA_VISIBLE_DEVICES=3 python CNNfeature2.py --database=LIVE-VQC
