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
import torch.nn as nn
from scipy import stats
import os.path
import pandas as pd
import skvideo.io

from Dataload import VQADataset, VQADataset_test
from torch.utils.data import Dataset

from GetFeatures import get_features
from torchvision import transforms, models
from PIL import Image
from models.network import Reg_Domain
import Sample

if __name__ == "__main__":
    parser = ArgumentParser(description='"Test Demo of VSFA')
    parser.add_argument('--model_path', default='/model/CVD2014-EXP2.pt', type=str,
                        help='model path')
    parser.add_argument('--video_path', default='/24918.mp4', type=str,
                        help='video path ')
    parser.add_argument('--video_format', default='RGB', type=str,
                        help='video format: RGB or YUV420 (default: RGB)')
    parser.add_argument('--video_width', type=int, default=None,
                        help='video width')
    parser.add_argument('--video_height', type=int, default=None,
                        help='video height')

    parser.add_argument('--frame_batch_size', type=int, default=32,
                        help='frame batch size for feature extraction (default: 32)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data preparation
    assert args.video_format == 'YUV420' or args.video_format == 'RGB'
    if args.video_format == 'YUV420':
        video_data = skvideo.io.vread(args.video_path, args.video_height, args.video_width, inputdict={'-pix_fmt': 'yuvj420p'})
    else:
        video_data = skvideo.io.vread(args.video_path)

    video_length = video_data.shape[0]
    video_channel = video_data.shape[3]
    video_height = video_data.shape[1]
    video_width = video_data.shape[2]
    transformed_video = torch.zeros([video_length, video_channel, video_height, video_width])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    num = Sample.ECO(video_data)
        
    transformed_video = torch.zeros([len(num), video_channel,  video_height, video_width])
    for i, frame_idx in enumerate(num):

        frame = video_data[frame_idx]

# 
        frame = Image.fromarray(frame)
        frame = transform(frame)
        transformed_video[i] = frame

    print('Video length: {}'.format(transformed_video.shape[0]))

    # feature extraction
    features = get_features(transformed_video)
    features = torch.unsqueeze(features.reshape(int(features.shape[0]), -1), 0)

    model = Reg_Domain(320,16, False)
    model.load_state_dict(torch.load(args.model_path))  #
    model.to(device)
    model.eval()
    with torch.no_grad():
        input_length = features.shape[1] * torch.ones(1, 1)
        outputs = model(features, input_length)
        y_pred = outputs.item()
        print("Predicted quality: {}".format(y_pred))
