# DualHash_NeurIPS

## Requirements
- Python 3.8.10
- PyTorch 1.11.0+cu113
- torchvision 0.12.0+cu113
- numpy 1.22.4
- tqdm 4.61.2
- Pillow 9.1.1
- CUDA support (recommended)

## Dataset Preparation
This project supports the following datasets:

### CIFAR-10
The CIFAR-10 dataset is included in the repository and will be automatically downloaded by the code.

> **Note:** We've configured the code to use datasets from the `data` folder by setting `cifar_dataset_root = 'data/cifar-10'` in `sgd_tools.py`. The code will look for index files at `data/cifar-10/train_index.npy`, `data/cifar-10/test_index.npy`, and `data/cifar-10/valid_index.npy`. This prevents redundant downloads and ensures consistent data paths.

### NUS-WIDE
Due to its large size, the NUS-WIDE dataset is not included in the repository. Please follow these steps:

1. Download the NUS-WIDE dataset from [Baidu Drive](https://pan.baidu.com/s/1f9mKXE2T8XpIq8p7y8Fa6Q) (Password: uhr3)
2. Extract and place only the image files in `data/nus-wide/images` directory

> **Note:** All necessary index files (`train_img.txt`, etc.) and label files are already included in the repository. Only image files need to be downloaded. To verify your setup, run `python DualHash.py --dataset nus-wide` - if no image errors occur, everything is properly prepared.

## Usage
``` python
python DualHash.py --dataset cifar-10
```
or

```**python**
python DualHash.py --dataset nus-wide
```
## Default Configuration

The project uses the following default configurations which can be found in the `get_config()` function:

### Model Parameters
- **Hidden Dimension**: 1024
- **Beta** (tanh scaling): 1.0
- **Network**: AlexNet (backbone)

### Training Parameters
- **Epochs**: 250
- **Batch Size**: 256 for CIFAR-10, 128 for NUS-WIDE
- **Learning Rate**: 0.01 initial with step decay
- **Step LR Decay**: Gamma factor 0.5-0.7
- **Save Epoch Start**: 70

### Loss Function Parameters
- **Alpha** (sigmoid scaling): 0.1
- **Alpha1** (step size for B): 1e-2
- **Alpha2** (step size for lambda): 1e-3
- **Lambda** (regularization): 5e-2
- **Eta** (penalty coefficient): 3-10 (CIFAR-10) / 0.1-5 (NUS-WIDE)
### Optimizer Parameters
#### SGD (DualHash)****
- **Momentum**: 0.905
- **Weight Decay**: 5e-4
- **Nesterov**: True

#### STORM (DualHash_storm)
- **Weight Decay**: 5e-4
- **C** (recursive momentum control): [0.01,0.05,0.1,0.2] cross-validation
- **Use Constant C**: True

### Evaluation Parameters
- **CIFAR-10**: mAP@ALL, Precision@1000, Precision@r=2
- **NUS-WIDE**: mAP@5000, Precision@1000, Precision@r=2

### Modifying Configuration
You can modify these parameters in the `get_config()` function in either `DualHash.py` or `DualHash_storm.py`. 

The configuration is dataset-specific and will be automatically selected based on the `--dataset` argument.****
## EXPERIMENTS RESULTS
CNN model: Alexnet. Compute mean average precision(MAP).

cifar10: 10 classes, 5000 query images, 10000 training images.

nus-wide-tc21: 21 classes, 6300 query images, 14700 training images.

### DualHash-StoM

| bits | 16 | 32 | 48 | 64 |
| :---: | :---: | :---: | :---: | :---: |
| cifar10@ALL | 0.8215 | 0.8481 | 0.8534 | 0.8539 |
| nus-wide@5000 | 0.6339 | 0.7002 | 0.7248 | 0.7448 |

### DualHash-StoRM
| bits | 16 | 32 | 48 | 64 |****
| :---: | :---: | :---: | :---: | :---: |
| cifar10@ALL | 0.8037 | 0.8051 | 0.8168 | 0.8345 |
| nus-wide@5000 | 0.6485 | 0.6802 | 0.6951 | 0.6982 |

