## 1.INSTALLATION
- Ubuntu 18.04.5 LTS
- CUDA 11.2
- Python 3.7.5
- Training PC: 1x RTX3090 (or any GPU with at least 24Gb VRAM), 32GB RAM.
- python packages are detailed separately in requirements.txt
```
$ conda create -n envs python=3.7.5
$ conda activate envs
$ pip install -r requirements.txt
$ pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
## 2.DATA
* Kalapa dataset.  
* Vietnamese address dataset `https://github.com/thien0291/vietnam_dataset`  
* Vietnamese poems corpus: `https://huggingface.co/datasets/phamson02/vietnamese-poetry-corpus`  
* Synthetic data. Download my generated data here (https://drive.google.com/file/d/1FfjVZqNRZGExZjZmqzWk_L3QDQS-20Bl/view?usp=sharing)  
* Or following the commands below to re-generating it. (make sure poems_dataset.csv and vietnam_dataset are inside synthetic_data/)    
```
$ cd synthetic_data  
$ python gendata_address.py    
$ python gendata_aug.py  
$ python gendata_poems.py  
$ cd ..  
$ python prepare_ext_data.py  
``` 

* Folder structure before executing training  
├── training_data   
│ ├── images    
│ ├── annotations    
├── synthetic_data   
│ ├── address    
│ ├── aug    
│ ├── poems  
│ ├── ...  
├── configs   
│ ├── b2_256_ptr_f5.py   
│ ├── b1_384_ptr_f5.py   
│ ├── ...   
├── train.py  
├── prepare_ext_data.py  
├── train_ext3.csv  
├── train_folds.csv  
├── ...  

## 3.TRAINING
* Pretrained models on synthetic data.  
```
$ python train.py -C b2_256_ptr_f5  
$ python train.py -C b1_384_ptr_f5  
```

* Fine-tune models on real data.  
```
$ python train.py -C b1_384_f5  
$ python train.py -C b2_256_f5  
```

## 4.INFERENCE

* Refer to submitted notebook

