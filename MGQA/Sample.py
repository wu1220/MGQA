from numpy.random import randint
import torch
import torch
from numpy.random import randint
import numpy as np

# TSN
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

# TSM
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



# ECO
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


# 5*5_32*32_patch
def get_spatial_fragments_5_5(
    video,
    fragments_h=5,
    fragments_w=5,
    fsize_h=32,
    fsize_w=32,
    aligned=32,
    nfrags=1,
    random=False,
    fallback_type="upsample",
):
    size_h = fragments_h * fsize_h
    size_w = fragments_w * fsize_w

    video = video.unsqueeze(1)

    if video.shape[1] == 1:
        aligned = 1

    dur_t, res_h, res_w = video.shape[-3:]
    ratio = min(res_h / size_h, res_w / size_w)
    if fallback_type == "upsample" and ratio < 1:

        ovideo = video
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=1 / ratio, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)

    assert dur_t % aligned == 0, "Please provide match vclip and align index"
    size = size_h, size_w

    ## make sure that sampling will not run out of the picture
    hgrids = torch.LongTensor(
        [min(res_h // fragments_h * i, res_h - fsize_h) for i in range(fragments_h)]
    )
    wgrids = torch.LongTensor(
        [min(res_w // fragments_w * i, res_w - fsize_w) for i in range(fragments_w)]
    )
    hlength, wlength = res_h // fragments_h, res_w // fragments_w

    if random:
        print("This part is deprecated. Please remind that.")
        if res_h > fsize_h:
            rnd_h = torch.randint(
                res_h - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if res_w > fsize_w:
            rnd_w = torch.randint(
                res_w - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
    else:
        if hlength > fsize_h:
            rnd_h = torch.randint(
                hlength - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if wlength > fsize_w:
            rnd_w = torch.randint(
                wlength - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()

    ten_num =  torch.zeros([fragments_h*fragments_w, 3,fsize_w,fsize_h])
    # target_videos = []
    target_video = torch.zeros(3,1,160,160).to(video.device)
    m = 0
    for i, hs in enumerate(hgrids):
        for j, ws in enumerate(wgrids):
            for t in range(dur_t // aligned):
                t_s, t_e = t * aligned, (t + 1) * aligned
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w
                if random:
                    h_so, h_eo = rnd_h[i][j][t], rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = rnd_w[i][j][t], rnd_w[i][j][t] + fsize_w
                else:
                    h_so, h_eo = hs + rnd_h[i][j][t], hs + rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = ws + rnd_w[i][j][t], ws + rnd_w[i][j][t] + fsize_w
                # if i != 0 and j != 0 and i !=4 and j != 4:
                    target_video[:, t_s:t_e, h_s:h_e, w_s:w_e] = video[
                        :, t_s:t_e, h_so:h_eo, w_so:w_eo
                    ]
                ten_num[m] = video[:, t_s:t_e, h_so:h_eo, w_so:w_eo].permute(1,0,2,3)
                m = m+1

    return target_video.permute(1,0,2,3)
