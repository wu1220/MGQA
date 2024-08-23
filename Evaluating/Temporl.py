import cv2
import os
import numpy as np
from numpy.random import randint


def ECO(video_data):
    N  = 20 
    tol_frames = video_data.shape[0]
    print(tol_frames)
    d =  tol_frames % N
    num = []
    if d == 0:
        l = int(tol_frames / N)
        for i in range(l):
            x = randint(i*N, (i+1)*N)
            num.append(x)
    else:
        l = int(tol_frames / 20)
        for i in range(l):
            x = randint(i*N, (i+1)*N)
            num.append(x)
        x = randint(tol_frames-N, tol_frames)
        num.append(x)
    
    return num


def TSN(video_data):
    
    vid = [1,2,3,4,5,6,7,8, 9,10]
    num = []

    tol_frames = video_data.shape[0]
    k = 4     # 超参数
    frames = int(tol_frames / len(vid))   #  每一个片段的帧数
    step = int((frames-1) / (k-1))   # 一个判断取关键帧位置参数
    # print(tol_frames)
    for i in (vid):
        
        if step>0:    # 判断step>0
            frame_tick = range(1,min((2+step*(k-1)),frames+1),step)  # 生成一个列表，从1到（2+step*(k-1),frames+1）其中最小值，跨度为step。
        else:
            frame_tick = [1] * k      # 否则生成一个全1是长度是四的列表
        assert(len(frame_tick)==k)
        for j in range(len(frame_tick)):   # 循环
        # print(frame_tick[j])
            if (frame_tick[j]+(frames*(i-1))) <tol_frames:
                num.append(frame_tick[j]+(frames*(i-1)))  # 得到每一个片段的帧
            else:
                num.append(frame_tick[j]+(frames*(i-1))-1) 
    return num

def TSM(video_data):
    video_num = 10  
    # 拆分为10个片段
    num = []

    tol_frames = video_data.shape[0]
    # print(tol_frames)

    average_duration = (tol_frames - 1 + 1) // video_num
    offsets = np.multiply(list(range(video_num)), average_duration) + randint(average_duration,  size=video_num)
    for seg_ind in offsets:
        p = int(seg_ind)
        for i in range(1):
            num.append(p)
            if p < tol_frames:
                p += 1

    return num


if __name__ == '__main__':
    num  = ECO('/media/data/wl/VQA_Model/VSFA/test.mp4')
    print(num)