# 2PESNet: Towards online processing of temporal action localization 
**2PESNet: Towards online processing of temporal action localization** (Pattern Recognition)   
Young Hwi Kim, Seonghyeon Nam, Seon Joo Kim   
[[`link`](https://doi.org/10.1016/j.patcog.2022.108871)]   

## Updates
**17 Jul, 2022**: Initial release

## Installation

### Prerequisites
- Ubuntu 18.04  
- Python 2.7.12   
- CUDA 10.0  

### Requirements
- pytorch==1.4.0  
- numpy==1.16.2
- h5py==2.9.0
- tensorboardX==2.1



## Training

### Input Features
We provide the Kinetics pre-trained feature of THUMOS'14 dataset. The extracted features can be downloaded from [here](https://drive.google.com/file/d/1GwQjMq0Eyc3XWljeeaSqwbTal5y76Xwy/view?usp=sharing).  
Files should be located in 'data/'.   

### Trained Model
The trained models that used Kinetics pre-trained feature can be downloaded from [link will be added soon].    
Files should be located in 'checkpoint/'. 

### Training Model by own
Training multi-head detector (MHD) 
```
python main.py --module=mhd --mode=train
```
Training end detection refinement (EDR)
```
python main.py --module=mhd --mode=data_gen
python main.py --module=edr --mode=train
```
Training action start detection (ASD)
```
python main.py --module=asd --mode=train
```


## Testing
Testing online performance of 2PESNet.
```
python main.py --module=asd --mode=test 
```

Testing online performance of 2PESNet with temporal tolerance.
```
python main.py --module=asd --mode=test --allowed_delay=2
```

## Paper Results

| Method | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:| 
| 2PESNet (T=0) | 44.2 | 36.9 | 28.7 | 19.9 | 12.1 |
| 2PESNet (T=2) | 47.4 | 39.8 | 31.4 | 21.8 | 14.0 |


## Updated Results
By the minor bug fix, we report the updated results.

| Method | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:| 
| 2PESNet (T=0) | 47.0 | 39.5 | 30.4 | 21.7 | 13.0 |
| 2PESNet (T=2) | 47.9 | 40.4 | 31.1 | 22.1 | 13.9 |



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
