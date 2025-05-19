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

```python
python DualHash.py --dataset nus-wide
```