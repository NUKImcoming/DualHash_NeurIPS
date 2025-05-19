# DualHash_NeurIPS

## Requirements
- Python 3.8.10
- PyTorch 1.11.0+cu113
****
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
Due to its large size, the NUS-WIDE dataset is not included in the repository. Please follow these steps to prepare the dataset:

1. Download the NUS-WIDE dataset from the [official website](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html)
2. Extract the downloaded files and place the image files in the `data/nus-wide/images` directory
3. Ensure that the label files are in the correct location

## Usage
``` python
python DualHash.py --dataset cifar-10
```
or

```python
python DualHash.py --dataset nus-wide
```