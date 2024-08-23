import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
import skvideo.io
from PIL import Image
import os
import Sample
import numpy as np

def getpath(video_dir, video_name):
    str = ['Color', 'Artifacts','Exposure','Focus', 'Sharpness', 'Stabilization']
    f = 0
    for s in str:
        path = video_dir+s
        for file in os.listdir(path):
            if file == video_name:
                P = path
                f = 1
                break
        if f == 1:
            break
    return P



class VideoDataset(Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, videos_dir, video_names, score, video_format='RGB', width=None, height=None):

        super(VideoDataset, self).__init__()
        self.videos_dir = videos_dir
        self.video_names = [str(video_name)+'.mp4' for video_name in video_names]
        self.score = score
        self.format = video_format
        self.width = 1080
        self.height = 1920

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        assert self.format == 'YUV420' or self.format == 'RGB'
        if self.format == 'YUV420':
            # video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name), self.height, self.width, inputdict={'-pix_fmt':'yuvj420p'})
            video_data = skvideo.io.vread(os.path.join(getpath(self.videos_dir, video_name), video_name), self.height, self.width, inputdict={'-pix_fmt':'yuvj420p'})
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
        
        # 时间采样
        num = Sample.ECO(video_data)
        
        transformed_video = torch.zeros([len(num), video_channel,  video_height, video_width])
        for i, frame_idx in enumerate(num):

            frame = video_data[frame_idx]

# 
            frame = Image.fromarray(frame)
            frame = transform(frame)
            transformed_video[i] = frame

        sample =  {'name':video_name,'video':transformed_video,'score': video_score}

        return sample

class VQADataset(Dataset):
    def __init__(self, features_dir, index=None, max_len=240, feat_dim=4096, scale=1):
        super(VQADataset, self).__init__()
        self.features = np.zeros((len(index), max_len, 1000))
        self.length = np.zeros((len(index), 1))
        self.mos = np.zeros((len(index), 1))
        for i in range(len(index)):
            # if index[i] != 163:
                features = np.load(features_dir + str(index[i]) +'.mp4'+ '_resnet-50_res5c.npy')   # [49,2049]
                # if features.shape[0] < 60:
                self.length[i] = features.shape[0]
                # print(i)
                # print(features.shape[0])
                self.features[i, :features.shape[0],:] = features.reshape(int(features.shape[0]), -1)[:,:1000]
                # self.features[i, :features.shape[0],:] = features[:,0,:,:].reshape(int(features.shape[0]), -1)
                self.mos[i] = np.load(features_dir + str(index[i]) +'.mp4'+ '_score.npy')  #
                
        self.scale = scale  #
        self.label = self.mos / self.scale  # label normalization

    def __len__(self):
        return len(self.mos)

    def __getitem__(self, idx):
        sample = self.features[idx], self.length[idx], self.label[idx]
        return sample

class VQADataset_test(Dataset):
    def __init__(self, features_dir, index=None, max_len=240, feat_dim=4096, scale=1):
        super(VQADataset_test, self).__init__()
        self.name = np.zeros((len(index), 1))
        self.name=[]
        self.features = np.zeros((len(index), max_len, 8000))
        self.length = np.zeros((len(index), 1))
        self.mos = np.zeros((len(index), 1))
        for i in range(len(index)):
                features = np.load(features_dir + str(index[i]) +'.mp4'+ '_resnet-50_res5c.npy')   # [49,2049]
                self.length[i] = features.shape[0]

                self.features[i, :features.shape[0],:] = features.reshape(int(features.shape[0]), -1)
                self.mos[i] = np.load(features_dir + str(index[i])+'.mp4' + '_score.npy')  #

        self.scale = scale  #
        self.label = self.mos / self.scale  # label normalization

    def __len__(self):
        return len(self.mos)

    def __getitem__(self, idx):
        sample = self.features[idx], self.length[idx], self.label[idx],self.name[idx]
        return sample
