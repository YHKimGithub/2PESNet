# 2PESNet: Towards online processing of temporal action localization 
**2PESNet: Towards online processing of temporal action localization** (Pattern Recognition)   
Young Hwi Kim, Seonghyeon Nam, Seon Joo Kim   
[[`link`]()]   

## Updates
**17 Jul, 2022**: Initial release

## Installation

### Prerequisites
- Ubuntu 16.04  
- Python 2.7.12   
- CUDA 10.0  

### Requirements
- pytorch==1.4.0  
- numpy==1.16.2
- h5py==2.9.0
- tensorboardX==2.1



## Training

### Input Features
We provide the Kinetics pre-trained feature of THUMOS'14 dataset.
The extracted features can be downloaded from [link will be added soon].   
Files should be located in 'data/'.  
You can also get the feature files from [here](https://github.com/wangxiang1230/OadTR).

### Trained Model
The trained models that used Kinetics pre-trained feature can be downloaded from [link will be added soon].    
Files should be located in 'checkpoints/'. 

### Training Model by own
To train the main OAT model, execute the command below.
```
python main.py --mode=train
```
To train the post-processing network (OSN), execute the commands below.
```
python supnet.py --mode=make --inference_subset=train
python supnet.py --mode=make --inference_subset=test
python supnet.py --mode=train
```


## Testing
To test OAT-OSN, execute the command below.
```
python main.py --mode=test
```

To test OAT-NMS, execute the command below.
```
python main.py --mode=test --pptype=nms
```

## Paper Results

| Method | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:| 
| 2PESNet (T=0) | 44.2 | 36.9 | 28.7 | 19.9 | 12.1 |
| 2PESNet (T=2) | 47.4 | 39.8 | 31.4 | 21.8 | 14.0 |


## Updated Results

| Method | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:| 
| 2PESNet (T=0) | 44.8 | 36.9 | 29.0 | 20.6 | 13.6 |
| 2PESNet (T=2) | 47.3 | 39.3 | 30.5 | 21.7 | 14.2 |



## Citing 2PESNet
Please cite our paper in your publications if it helps your research:

```BibTeX
@article{KIM2022108871,
title = {2PESNet: Towards online processing of temporal action localization},
journal = {Pattern Recognition},
volume = {131},
pages = {108871},
year = {2022},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2022.108871},
url = {https://www.sciencedirect.com/science/article/pii/S0031320322003521},
author = {Kim, Young Hwi and Nam, Seonghyeon and Kim, Seon Joo}
}
```
