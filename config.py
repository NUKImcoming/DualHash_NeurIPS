import os
import torch.optim as optim
import torch
import numpy as np
import random

def get_base_config(dataset_name, optimizer="sgdm", info="DualHash", network_choice="alexnet"):
    """Get base configuration including dataset-specific and common settings"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # 网络配置选项
    network_configs = {
        "alexnet": {
            "net_class": "AlexNet",
            "info": "MDSHC_AlexNet",
            "feature_lr": 0.001,
            "hash_lr": 0.01,
            "hidden_dim": 4096,  # AlexNet隐藏层维度
            "beta": 0.1,
        },
        "resnet50": {
            "net_class": "ResNet50",
            "info": "MDSHC_ResNet50", 
            "feature_lr": 0.0001,
            "hash_lr": 0.001,
            "hidden_dim": 2048,  # ResNet50隐藏层维度
            "beta": 0.1,
        }
    }
    
    if network_choice not in network_configs:
        raise ValueError(f"Unsupported network: {network_choice}")
    
    network_config = network_configs[network_choice]
    
    dataset_configs = {
        "cifar-10": {
            "dataset": "cifar-10",
            "n_class": 10,
            "resize_size": 256,
            "crop_size": 224,
            "batch_size": 256,
            "save_path": os.path.join(project_root, "results", "save","CIFAR10"),
            "log_dir": os.path.join(project_root, "results", "logs", "CIFAR10"),
            "topK": 1000,
            "topK_mAP": -1,
            "r": 2,
            "data_path": "./data/cifar-10/images/",
        },
        "nus-wide": {
            "dataset": "nus-wide",
            "n_class": 21,
            "resize_size": 256,
            "crop_size": 224,
            "batch_size": 256,
            "save_path": os.path.join(project_root, "results", "save", "NUSWIDE"),
            "log_dir": os.path.join(project_root, "results", "logs",  "NUSWIDE"),
            "topK": 1000,
            "topK_mAP": 5000,  
            "r": 2,
            "data_path": "./data/imagenet-100/images/",
        },
        "imagenet100": {
            "dataset": "imagenet100",
            "n_class": 100,
            "resize_size": 256,
            "crop_size": 224,
            "batch_size": 256,
            "save_path": os.path.join(project_root, "results", "save", "IMAGENET100"),
            "log_dir": os.path.join(project_root, "results", "logs",   "IMAGENET100"),
            "topK": 1000,
            "topK_mAP": 1000,
            "r": 2,
            "data_path": "./data/imagenet-100/images/",
        }
    }
    
    if dataset_name not in dataset_configs:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    base_config = dataset_configs[dataset_name]
    
    # 合并网络配置
    base_config.update(network_config)
    
    # Add common configuration
    base_config.update({
        # Training parameters
        "epoch": 300,
        "save_epoch_start": 70,
        "step_num": 10, 
        "eval_epoch_interval": 5, 
        "log_epoch_interval": 10,
        
        # Hardware parameters
        "device": torch.device("mps"),
        "bit_list": [64, 48, 32, 16],
        "info": info,
        "save_path": os.path.join(base_config["save_path"], info),
        "log_dir": os.path.join(base_config["log_dir"], info),
        
        # Random seed for reproducibility
        "seed": 2024,
    })

    # Add optimizer configuration
    if optimizer == "sgdm":
        optimizer_config = {
            "type": optim.SGD, 
            "optim_params": {
                "lr": 0.01,  
                "momentum": 0.905, 
                "weight_decay": 5e-4, 
                "nesterov": True
            },
            "lr_type": "step",  
            "lr_param": {
                "init_lr": 0.1, 
                "gamma": 0.5,
                "step": None
            }
        }
    elif optimizer == "storm":
        optimizer_config = {
            "type": optim.SGD, 
            "optim_params": {
                "lr": 0.01,  
                "momentum": 0.9, 
                "weight_decay": 5e-4, 
                "nesterov": False
            },
            "lr_type": "step",  
            "lr_param": {
                "init_lr": 0.1, 
                "gamma": 0.5,
                "step": None
            }
        }
    
    base_config["optimizer"] = optimizer_config

    return base_config

    """Get base configuration including dataset-specific and common settings"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    dataset_configs = {
        "cifar-10": {
            "dataset": "cifar-10",
            "n_class": 10,
            "resize_size": 256,
            "crop_size": 224,
            "batch_size": 256,
            "save_path": os.path.join(project_root, "results", "save","CIFAR10"),
            "log_dir": os.path.join(project_root, "results", "logs", "CIFAR10"),
            "topK": 1000,
            "topK_mAP": -1,
            "r": 2,
            "data_path": "./data/cifar-10/images/",
        },
        "nus-wide": {
            "dataset": "nus-wide",
            "n_class": 21,
            "resize_size": 256,
            "crop_size": 224,
            "batch_size": 256,
            "save_path": os.path.join(project_root, "results", "save", "NUSWIDE"),
            "log_dir": os.path.join(project_root, "results", "logs",  "NUSWIDE"),
            "topK": 1000,
            "topK_mAP": 5000,  
            "r": 2,
            "data_path": "./data/imagenet-100/images/",
        },
        "imagenet100": {
            "dataset": "imagenet100",
            "n_class": 100,
            "resize_size": 256,
            "crop_size": 224,
            "batch_size": 256,
            "save_path": os.path.join(project_root, "results", "save", "IMAGENET100"),
            "log_dir": os.path.join(project_root, "results", "logs",   "IMAGENET100"),
            "topK": 1000,
            "topK_mAP": 1000,
            "r": 2,
            "data_path": "./data/imagenet-100/images/",
        }
    }
    
    if dataset_name not in dataset_configs:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    base_config = dataset_configs[dataset_name]
    
    # Add common configuration
    base_config.update({
        # Training parameters
        "epoch": 300,
        "save_epoch_start": 70,
        "step_num": 10, 
        "eval_epoch_interval": 5, 
        "log_epoch_interval": 10,
        
        # Hardware parameters
        "device": torch.device("mps"),  # or "cuda:0" if torch.cuda.is_available() else "cpu"
        "bit_list": [64, 48, 32, 16],
        "info": info,
        "save_path": os.path.join(base_config["save_path"], info),
        "log_dir": os.path.join(base_config["log_dir"], info),
        
        # Random seed for reproducibility
        "seed": 2024,
    })

    # Add optimizer configuration
    if optimizer == "sgdm":
        optimizer_config = {
            "type": optim.SGD, 
            "optim_params": {
                "lr": 0.01,  
                "momentum": 0.905, 
                "weight_decay": 5e-4, 
                "nesterov": True
            },
            "lr_type": "step",  
            "lr_param": {
                "init_lr": 0.1, 
                "gamma": 0.5,
                "step": None
            }
        }
    elif optimizer == "storm":
        optimizer_config = {
            "type": optim.SGD, 
            "optim_params": {
                "lr": 0.01,  
                "momentum": 0.9, 
                "weight_decay": 5e-4, 
                "nesterov": False
            },
            "lr_type": "step",  
            "lr_param": {
                "init_lr": 0.1, 
                "gamma": 0.5,
                "step": None
            }
        }
    
    base_config["optimizer"] = optimizer_config
    
     # 网络配置选项
    network_configs = {
        "alexnet": {
            "net_class": "AlexNet",  # 这里应该是网络类名
            "info": "MDSHC_AlexNet",
            "feature_lr": 0.001,
            "hash_lr": 0.01,
        },
        "resnet50": {
            "net_class": "ResNet50",  # 这里应该是网络类名
            "info": "MDSHC_ResNet50", 
            "feature_lr": 0.0001,  # ResNet需要更小的学习率
            "hash_lr": 0.001,
        }
    }
    

    if network_choice not in network_configs:
        raise ValueError(f"Unsupported network: {network_choice}")
    
    network_config = network_configs[network_choice]

    return base_config


def get_storm_loss_config():
    """Get Storm-specific loss configuration"""
    return {
        "alpha": 0.1,
        "lambda": 1e-3,
        "eta": 1,
    }


def create_storm_config(dataset_name):
    """Create Storm configuration"""
    base_config = get_base_config(dataset_name, optimizer="storm", info="Storm")
    loss_config = get_storm_loss_config()
    
    config = {
        **base_config,
        **loss_config,
    }
    
    os.makedirs(config["save_path"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)
    
    return config

# For backward compatibility
def create_config(dataset_name):
    """Alias for create_dualhash_config for backward compatibility"""
    return create_dualhash_config(dataset_name) 

def setup_seed(seed):
    """设置随机种子以确保结果可复现"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False  # 为了速度通常设为 False
        torch.backends.cudnn.benchmark = True
    print(f"随机种子已设置为 {seed}")


def create_dualhash_config(dataset_name):
    """Create DualHash configuration"""
    base_config = get_base_config(dataset_name, optimizer="sgdm", info="DualHash")
    
    # DualHash-specific loss configuration
    loss_config = {
        "alpha": 0.1,
        "alpha1": 1e-2,  # stepsize for B
        "alpha2": 1e-3,  # stepsize for lambda
        "lambda": 5e-2,  # lambda for W-type 
        "eta": 1,  # penalty coefficent
        "dcc": 1,
    }
    
    config = {
        **base_config,
        **loss_config,
    }
    
    os.makedirs(config["save_path"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)
    
    return config 
