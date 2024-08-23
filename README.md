# Video Quality Assessment for Online Processing: From Spatial to Temporal Sampling

## Description
code for the following papers:

- Jiebin Yan, Lei Wu, Yuming Fang, Xuelin Liu, Xue Xia, Weide Liu. [Video Quality Assessment for Online Processing: From Spatial to Temporal Sampling](链接). 期刊名
![Framework](https://github.com/wu1220/MGQA/blob/main/motive1.png)

### Evaluating
In this part, the training and testing code is the same as VSFA(https://github.com/lidq92/VSFA), we only need to change the preprocessing part.


### Intra-Database Experiments (Training and Evaluating)
#### Feature extraction

```
CUDA_VISIBLE_DEVICES=0 python CNNfeatures.py --database=KoNViD-1k --frame_batch_size=64
```

You need to specify the `database` and change the corresponding `videos_dir`.

#### Quality prediction

```
CUDA_VISIBLE_DEVICES=0 python VSFA.py --database=KoNViD-1k --exp_id=0
```

You need to specify the `database` and `exp_id`.


### Test Demo

The model weights provided in `models/VSFA.pt` are the saved weights when running the 9-th split of KoNViD-1k.
```
python test_demo.py --video_path=test.mp4
```

### Requirement
```bash
conda create -n MGQA pip python=3.8
source activate MGQA
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
source deactive
```
- PyTorch 1.1.0

Note: The codes can also be directly run on PyTorch 1.3.


### Licence
You can use, redistribute, and adapt the material for non-commercial purposes, as long as you give appropriate credit by citing our paper and indicate any changes that you've made.
