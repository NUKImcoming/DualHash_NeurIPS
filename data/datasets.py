

import numpy as np
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as dsets
import os
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

# ========================================
# CIFAR-10 Dataset
# ========================================
class MyCIFAR10(dsets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        target = np.eye(10, dtype=np.int8)[np.array(target)]
        return img, target, index

def cifar_dataset(config):
    """Load CIFAR-10 dataset with custom train/valid/test split"""

    cifar_dataset_root = './data/cifar-10'
    
    batch_size = config["batch_size"]
    train_size = 1000
    test_size = 500
    valid_size = 500 

    transform = transforms.Compose([
        transforms.Resize(config["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = MyCIFAR10(root=cifar_dataset_root, train=True, transform=transform, download=True)
    test_dataset = MyCIFAR10(root=cifar_dataset_root, train=False, transform=transform)
    valid_dataset = MyCIFAR10(root=cifar_dataset_root, train=True, transform=transform, download=True)
    
    X = np.concatenate((train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))

    if os.path.exists(f'{cifar_dataset_root}/train_index.npy') and os.path.exists(f'{cifar_dataset_root}/test_index.npy') and os.path.exists(f'{cifar_dataset_root}/valid_index.npy'):
        print("Load saved indices!")
        train_index = np.load(f'{cifar_dataset_root}/train_index.npy')
        test_index = np.load(f'{cifar_dataset_root}/test_index.npy')
        valid_index = np.load(f'{cifar_dataset_root}/valid_index.npy')
    else:
        print("First Load indices!")
        train_index = []
        test_index = []
        valid_index = []

        for label in range(10):
            index = np.where(L == label)[0]
            np.random.shuffle(index)
            train_index.extend(index[:train_size])
            valid_index.extend(index[train_size:train_size + valid_size])
            test_index.extend(index[train_size + valid_size:train_size + valid_size + test_size])

        train_index = np.array(train_index)
        test_index = np.array(test_index)
        valid_index = np.array(valid_index)

        np.save(f'{cifar_dataset_root}/train_index.npy', train_index)
        np.save(f'{cifar_dataset_root}/test_index.npy', test_index)
        np.save(f'{cifar_dataset_root}/valid_index.npy', valid_index)

    train_dataset.data = X[train_index]
    train_dataset.targets = L[train_index]
    test_dataset.data = X[test_index]
    test_dataset.targets = L[test_index]
    valid_dataset.data = X[valid_index]
    valid_dataset.targets = L[valid_index]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader, valid_loader, train_index.shape[0], test_index.shape[0], valid_index.shape[0]

# ========================================
# NUS-WIDE Dataset
# ========================================
def train_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),                         
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

def query_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

class NusWideDatasetTC21(Dataset):
    def __init__(self, root, img_txt, label_txt, transform=None, train=None):
        self.root = root
        self.transform = transform

        img_txt_path = os.path.join(root, img_txt)
        label_txt_path = os.path.join(root, label_txt)

        with open(img_txt_path, 'r') as f:
            self.data = np.array([i.strip() for i in f])
        self.targets = np.loadtxt(label_txt_path, dtype=np.float32)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.data[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[index], index

    def __len__(self):
        return len(self.data)

    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()

def nus_wide_dataset(config):
    """Load NUS-WIDE dataset with custom train/valid/test split"""
    batch_size = config["batch_size"]
    
    nus_wide_root = './data/nus-wide'
    
    # Check if images exist
    image_dir = os.path.join(nus_wide_root, 'images')
    if not os.path.exists(image_dir):
        try:
            os.makedirs(image_dir, exist_ok=True)
            print(f"Created images directory at {image_dir}")
        except Exception as e:
            print(f"Error creating directory: {e}")
    
    sample_img_path = None
    try:
        with open(f'{nus_wide_root}/train_img.txt', 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                sample_img_path = os.path.join(nus_wide_root, first_line)
    except Exception as e:
        print(f"Error reading image index file: {e}")
    
    images_exist = False
    if sample_img_path and os.path.exists(sample_img_path):
        images_exist = True
    
    if not images_exist:
        print("\n" + "="*80)
        print("WARNING: NUS-WIDE dataset images not found!")
        print("The code will attempt to run but will fail when trying to load images.")
        print("\nTo use the NUS-WIDE dataset, please:")
        print('1. Download the dataset from: "https://pan.baidu.com/s/1f9mKXE2T8XpIq8p7y8Fa6Q"')
        print("2. Extract images to: data/nus-wide/images/")
        print("3. Ensure image paths in txt files match your directory structure")
        print("="*80 + "\n")
        print("Continuing with NUS-WIDE dataset setup (will fail when loading images)...")

    # Load datasets
    train_dataset = NusWideDatasetTC21(
        root=nus_wide_root,
        img_txt='train_img.txt',
        label_txt='train_label_onehot.txt',
        transform=train_transform(),
        train=True
    )
    
    valid_dataset = NusWideDatasetTC21(
        root=nus_wide_root,
        img_txt='valid_img.txt',
        label_txt='valid_label_onehot.txt',
        transform=train_transform(),
        train=True
    )
    
    test_dataset = NusWideDatasetTC21(
        root=nus_wide_root, 
        img_txt='test_img.txt',
        label_txt='test_label_onehot.txt',
        transform=query_transform(),
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True, num_workers=4)
    
    num_train = len(train_dataset.data)
    num_test = len(test_dataset.data)
    num_valid = len(valid_dataset.data)
    
    return train_dataloader, test_dataloader, valid_dataloader, num_train, num_test, num_valid

# ========================================
# ImageNet100 Dataset
# ========================================
class ImageList(object):
    """ImageNet100 Dataset Loader"""
    def __init__(self, data_path, image_list, transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)

def image_transform(resize_size, crop_size, data_set):
    """ImageNet100 image transformation"""
    if data_set == "train_set":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose([transforms.Resize(resize_size)]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])

def imagenet100_dataset(config):
    """Load ImageNet100 dataset with custom train/valid/test split"""
    batch_size = config["batch_size"]
    
    imagenet100_root = './data/imagenet-100'
    
    # Load data splits
    with open(f'{imagenet100_root}/train_img.txt', 'r') as f:
        train_lines = f.readlines()
    
    if os.path.exists(f'{imagenet100_root}/valid_img.txt'):
        with open(f'{imagenet100_root}/valid_img.txt', 'r') as f:
            val_lines = f.readlines()
    else:
        print("Warning: valid_img.txt not found, using test_img.txt as validation set")
        with open(f'{imagenet100_root}/test_img.txt', 'r') as f:
            val_lines = f.readlines()
    
    with open(f'{imagenet100_root}/test_img.txt', 'r') as f:
        test_lines = f.readlines()
    
    # Create transforms
    transform = image_transform(config["resize_size"], config["crop_size"], "train_set")
    test_transform = image_transform(config["resize_size"], config["crop_size"], "test")
    
    # Load datasets
    train_dataset = ImageList(config["data_path"], train_lines, transform)
    val_dataset = ImageList(config["data_path"], val_lines, test_transform)
    test_dataset = ImageList(config["data_path"], test_lines, test_transform)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Dataset loading completed:")
    print(f"  Training set (database): {len(train_dataset)} images")
    print(f"  Validation set (query): {len(val_dataset)} images")
    print(f"  Test set (query): {len(test_dataset)} images")
    
    return train_loader, test_loader, val_loader, len(train_dataset), len(test_dataset), len(val_dataset)

# ========================================
# Unified Dataset Interface
# ========================================
def get_data(config):
    """
    Unified dataset interface
    
    Args:
        config (dict): Dictionary containing dataset configuration, must include 'dataset' field
        
    Returns:
        tuple: (train_loader, test_loader, valid_loader, num_train, num_test, num_valid)
    """
    dataset_name = config.get("dataset", "").lower()
    
    if dataset_name == "cifar-10":
        return cifar_dataset(config)
    elif dataset_name == "nus-wide":
        return nus_wide_dataset(config)
    elif dataset_name == "imagenet100":
        return imagenet100_dataset(config)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: ['cifar-10', 'nus-wide', 'imagenet100']")