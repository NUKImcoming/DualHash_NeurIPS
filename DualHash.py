import numpy as np
import torch
import time
import torch.optim as optim
from network import AlexNet
from sgd_tools import *
import os
from Optimizer_B_lambda import *
import lr_schedule
from tqdm import tqdm
import math
from torch.utils.tensorboard import SummaryWriter

torch.multiprocessing.set_sharing_strategy('file_system')

seed = 2024 
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os

def get_config(dataset_choice="cifar-10"):
    # get the current script path (project root)
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # define different dataset parameters
    dataset_configs = {
        "cifar-10": {
            "dataset": "cifar-10",
            "n_class": 10,
            "resize_size": 256,
            "crop_size": 224,
            "batch_size": 256,
            "save_path": os.path.join(project_root, "results", "save", "STOM", "CIFAR10"),
            "log_dir": os.path.join(project_root, "results", "logs", "STOM", "CIFAR10"),
            "topK": 1000,
            "topK_mAP": -1,
            "r": 2,
        },
        "nus-wide": {
            "dataset": "nus-wide",
            "n_class": 21,
            "resize_size": 256,
            "crop_size": 224,
            "batch_size": 256,
            "save_path": os.path.join(project_root, "results", "save", "STOM", "NUSWIDE"),
            "log_dir": os.path.join(project_root, "results", "logs", "STOM", "NUSWIDE"),
            "topK": 1000,
            "topK_mAP": 5000,  
            "r": 2,
        }
    }
    
    # select the current dataset configuration
    dataset_config = dataset_configs[dataset_choice]
    
    # basic configuration
    config = {
        # model parameters
        "model_params": {
            "beta": 1,
            "hidden_dim": 1024,
            "net": AlexNet,
            "info": "DualHash",
        },
        
        # dataset parameters - using the selected dataset configuration
        "dataset_params": {
            "dataset": dataset_config["dataset"],
            "n_class": dataset_config["n_class"],
            "resize_size": dataset_config["resize_size"],
            "crop_size": dataset_config["crop_size"],
            "batch_size": dataset_config["batch_size"],
        },
        
        # optimizer parameters
        "optimizer": {
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
                "gamma": 0.5,  # 0.7,
                "step": None
            }
        },
        
        # loss function parameters
        "loss_params": {
            "alpha": 0.1,
            "alpha1": 1e-2,  # stepsize for B
            "alpha2": 1e-3,  # stepsize for lambda
            "lambda": 5e-2,  # lambda for W-type 
            "eta": 1,  # penalty coefficent
        },
        
        # training parameters
        "training_params": {
            "epoch": 300,
            "save_epoch_start": 70,
            "step_num": 10, 
            "dcc": 1,  # 1~5
            "log_interval": 10, 
        },
        
        # evaluation parameters
        "eval_params": {
            "topK": dataset_config["topK"],
            "topK_mAP": dataset_config["topK_mAP"],
            "r": dataset_config["r"],
        },
        
        # path parameters - using the selected dataset configuration
        "path_params": {
            "save_path": dataset_config["save_path"],
            "log_dir": dataset_config["log_dir"],
        },
        
        # hardware parameters
        # "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "device": torch.device("mps"), 
        
        # hash bit number
        "bit_list": [64,48,32,16],
    }
    
    # ensure the directory exists
    os.makedirs(config["path_params"]["save_path"], exist_ok=True)
    os.makedirs(config["path_params"]["log_dir"], exist_ok=True)
    
    # flatten the configuration
    flat_config = {}
    for category, params in config.items():
        if isinstance(params, dict) and category != "optimizer":
            for key, value in params.items():
                flat_config[key] = value
        else:
            flat_config[category] = params
    
    return flat_config

	
class DualHashLoss(torch.nn.Module):
    def __init__(self, config):
        super(DualHashLoss, self).__init__()
        
    def forward(self, u, y, ind, B, config):
        sigmoid_alpha = config["alpha"]

        inner_product = sigmoid_alpha * u @ u.t() * 0.5
        s = (y @ y.t() > 0).float()

        likelihood_loss = (1 + (-inner_product.abs()).exp()).log() + inner_product.clamp(min=0) - s * inner_product
        likelihood_loss = likelihood_loss.mean()

        quan_loss = (B[:, ind] - u.t().data).pow(2).mean()
        ncvx_regu_loss = (1 - B.abs()).abs().mean()
    
        loss = likelihood_loss + config["eta"] * quan_loss + config["lambda"] * ncvx_regu_loss
        return loss

def train_val(config, bit):
    log_dir = os.path.join(config["log_dir"], f"{config['dataset']}_{bit}bits")
    writer = SummaryWriter(log_dir=log_dir)
    
    device = config["device"]
    train_loader, test_loader, valid_loader, num_train, num_test, num_valid = get_data(config)
    config["num_train"] = num_train
    config["num_test"] = num_test
    config["num_valid"] = num_valid
    config["batch_num"] = num_train // config["batch_size"] + 1
    
    print(f"train_dataset: {num_train}")
    print(f"test_dataset: {num_test}")
    print(f"valid_dataset: {num_valid}")
    
    net = config["net"](config["info"], config["hidden_dim"], bit, config["beta"]).to(device)
    
    parameter_list = [
        {"params": net.feature_layers.parameters(), "lr": 1},
        {"params": net.hash_layers.parameters(), "lr": 10}
    ]
    
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, **(config["optimizer"]["optim_params"]))
    
    param_lr = []
    layers = ["feature_layers", "hash_layers"]
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    
    config["Network_init_lr"] = dict(zip(layers, param_lr))
    config["max_iter"] = config["epoch"] * config["batch_num"] 
    optimizer_config["lr_param"]["step"] = math.ceil(config["max_iter"] / config["step_num"])
    schedule_param = optimizer_config["lr_param"]
    print("Schedule Parameters:", schedule_param)
    lr_scheduler = lr_schedule.schedule_dict[config["optimizer"]["lr_type"]]
    # print(f"hashNet's learning rate adjustment strategy is lr_scheduler: {lr_scheduler}")
    
    print(config)
    criterion = DualHashLoss(config)
    
    B = initialize_B_with_ITQ(train_loader, net, bit, device)
    U = torch.zeros(bit, num_train).to(device)
    Z = torch.zeros(bit, num_train).to(device)
    
    Best_mAP = 0.0
    train_losses = []
    training_times = []
    # val
    maps = []
    ap_topKs = []
    ap_rs = []                     
    iter_num = 0
    decay_times = 0
    # test
    tst_mAP = 0.0
    tst_AP_r = 0.0
    tst_AP_topK = 0.0
    tst_results = []

    for epoch in range(config["epoch"]):
        print("-----------------training-----------------")
        start = time.time()
        net.train()
        train_loss = 0
        
        for image, label, ind in tqdm(train_loader, leave=True):
            image = image.to(device)
            label = label.to(device)
            
            decay_times, optimizer = lr_scheduler(param_lr, optimizer, iter_num, **schedule_param)
            
            optimizer.zero_grad()
            u = net(image)
            U[:, ind] = u.t().data
            loss = criterion(u, label.float(), ind, B, config)
            train_loss += loss.item()
            loss.backward() 
            optimizer.step()
            U[:, ind] = net(image).t().data
            iter_num += 1
        
        for dcc_iter in range(config["dcc"]):
            B_prime = B
            B = B - config["alpha1"] * (2 * config["eta"] * (B - U) + Z)
            Z = updateZ(Z, B, B_prime, config["lambda"], config["alpha2"])
            
        train_loss = train_loss / num_train
        train_losses.append(train_loss)
        end = time.time()
        duration = end - start
        training_times.append(duration)

        print("-----------------validating-----------------")
        val_binary, val_label = compute_result(valid_loader, net, device=device)
        print("query_codes_shape: ", val_binary.shape)
        print("val_label_shape: ", val_label.shape)
        
        trn_binary, trn_label = compute_result(train_loader, net, device=device)
        
        mAP = hash_ranking_map(trn_binary.cpu().numpy(), trn_label.cpu().numpy(), val_binary.cpu().numpy(), val_label.cpu().numpy(), topk=config["topK_mAP"], dataset_type=config["dataset"])
        AP_topK, _ = get_precision_recall_topK(trn_binary.cpu().numpy(), trn_label.cpu().numpy(), val_binary.cpu().numpy(), val_label.cpu().numpy(), topk=config["topK"])
        AP_r, _ = get_precision_recall_within_hamming_radius(trn_binary.cpu().numpy(), trn_label.cpu().numpy(), val_binary.cpu().numpy(), val_label.cpu().numpy(), r=config["r"])
        
        if (epoch + 1) % config["log_interval"] == 0:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('mAP/val', mAP, epoch)
            writer.add_scalar('AP_topK/val', AP_topK, epoch)
            writer.add_scalar('AP_r/val', AP_r, epoch)
        
        if mAP > Best_mAP:
            tst_results = []
            Best_mAP = mAP
            print("-----------------testing-----------------")
            tst_binary, tst_label = compute_result(test_loader, net, device=device)
            tst_mAP = hash_ranking_map(trn_binary.cpu().numpy(), trn_label.cpu().numpy(), tst_binary.cpu().numpy(), tst_label.cpu().numpy(), topk=config["topK_mAP"], dataset_type=config["dataset"])
            tst_results.append(tst_mAP)
            tst_AP_topK, _ = get_precision_recall_topK(trn_binary.cpu().numpy(), trn_label.cpu().numpy(), tst_binary.cpu().numpy(), tst_label.cpu().numpy(), topk=config["topK"])
            tst_results.append(tst_AP_topK)
            tst_AP_r, _ = get_precision_recall_within_hamming_radius(trn_binary.cpu().numpy(), trn_label.cpu().numpy(), tst_binary.cpu().numpy(), tst_label.cpu().numpy(), r=config["r"])
            tst_results.append(tst_AP_r)
            
            writer.add_scalar('mAP/tst', tst_mAP, epoch)
            writer.add_scalar('AP_topK/tst', tst_AP_topK, epoch)
            writer.add_scalar('AP_r/tst', tst_AP_r, epoch)

            results_path = os.path.join(config["save_path"], f"{config['dataset']}_{bit}bits")
            os.makedirs(results_path, exist_ok=True)
            results_file = os.path.join(results_path, "test_results.txt")
            
            if not os.path.exists(results_file):
                with open(results_file, 'w') as f:
                    f.write("tst_mAP tst_AP_topK tst_AP_r\n")

            with open(results_file, 'a') as f:
                f.write(f"{tst_mAP} {tst_AP_topK} {tst_AP_r}\n")

            if epoch >= config["save_epoch_start"]:
                best_model_path = os.path.join(config["save_path"], f'{config["dataset"]}_{bit}bits_{tst_mAP}')
                os.makedirs(best_model_path, exist_ok=True)
                np.save(os.path.join(best_model_path, "val_label.npy"), val_label.numpy())
                np.save(os.path.join(best_model_path, "val_binary.npy"), val_binary.numpy())
                np.save(os.path.join(best_model_path, "trn_binary.npy"), trn_binary.numpy())
                np.save(os.path.join(best_model_path, "trn_label.npy"), trn_label.numpy())
                np.save(os.path.join(best_model_path, "tst_binary.npy"), tst_binary.numpy())
                np.save(os.path.join(best_model_path, "tst_label.npy"), tst_label.numpy())
                np.save(os.path.join(best_model_path, "tst_results.npy"), tst_results)
                torch.save(net.state_dict(), os.path.join(best_model_path, "model.pt"))
                print("save test results successfully!")
            
        maps.append(mAP)
        ap_topKs.append(AP_topK)
        ap_rs.append(AP_r)

        print(f"End of epoch {epoch + 1}/{config['epoch']}, Total Iterations: {iter_num}/{config['max_iter']}, decay_times:{decay_times}, bit:{bit}, dataset:{config['dataset']}, AP_topK:{AP_topK:.4f}, AP_r:{AP_r:.4f}, mAP:{mAP:.4f}, Best mAP: {Best_mAP:.4f}, tst_AP_topK:{tst_AP_topK:.4f}, tst_mAP:{tst_mAP:.4f}, tst_AP_r:{tst_AP_r:.4f}, time: {duration:.4f}, train_loss:{train_loss:.4f}")

    save_path = os.path.join(config["save_path"], f"{config['dataset']}_{bit}bits")
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, "mAP.npy"), maps)
    np.save(os.path.join(save_path, "AP_topK.npy"), ap_topKs)
    np.save(os.path.join(save_path, "AP_r.npy"), ap_rs)
    np.save(os.path.join(save_path, "train_loss.npy"), train_losses)
    np.save(os.path.join(save_path, "time.npy"), training_times)
      
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar-10', 
                        choices=['cifar-10', 'nus-wide'],
                        help='Dataset to use: cifar-10 or nus-wide')
    args = parser.parse_args()
    
    config = get_config(args.dataset)
    print(f"Using dataset: {args.dataset}")

    for bit in config["bit_list"]:
        train_val(config, bit)
