import numpy as np
import torch
import time
import torch.optim as optim
from network import AlexNet
from sgd_tools import *
import os
import lr_schedule
from tqdm import tqdm
import math
from torch.utils.tensorboard import SummaryWriter

torch.multiprocessing.set_sharing_strategy('file_system')

# 设置随机种子
seed = 2024 
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

def get_config(dataset_choice="cifar-10"):
    """获取DTSH配置"""
    project_root = "/root/autodl-tmp/save/"  # 参考DSH的路径
    
    dataset_configs = {
        "cifar-10": {
            "dataset": "cifar-10",
            "n_class": 10,
            "resize_size": 256,
            "crop_size": 224,
            "batch_size": 256,
            "save_path": os.path.join(project_root, "DTSH", "AlexNet", "CIFAR10"),
            "log_dir": os.path.join("/root/tf-logs/", "DTSH", "AlexNet", "CIFAR10"),
            "topK": 1000,
            "topK_mAP": -1,
            "r": 2,
        },
        "nus-wide": {
            "dataset": "nus-wide",
            "n_class": 21,
            "resize_size": 256,
            "crop_size": 224,
            "batch_size": 128,
            "save_path": os.path.join(project_root, "DTSH", "AlexNet", "NUSWIDE"),
            "log_dir": os.path.join("/root/tf-logs/", "DTSH", "AlexNet", "NUSWIDE"),
            "topK": 1000,
            "topK_mAP": 5000,  
            "r": 2,
        }
    }
    
    dataset_config = dataset_configs[dataset_choice]
    
    config = {
        # 模型参数
        "model_params": {
            "beta": 1,
            "hidden_dim": 1024,
            "net": AlexNet,
            "info": "DTSH_AlexNet",
        },
        
        # 数据集参数
        "dataset_params": {
            "dataset": dataset_config["dataset"],
            "n_class": dataset_config["n_class"],
            "resize_size": dataset_config["resize_size"],
            "crop_size": dataset_config["crop_size"],
            "batch_size": dataset_config["batch_size"],
        },
        
        # 优化器参数
        "optimizer": {
            "type": optim.SGD,  
            "optim_params": {
                "lr": 1e-3,  # DTSH使用较小的学习率
                "weight_decay": 1e-5
            },
            "lr_type": "step",
            "lr_param": {
                "init_lr": 1e-3,
                "gamma": 0.7,
                "step": None
            }
        },
        
        # DTSH特定参数
        "dtsh_params": {
            "alpha": 10,      # triplet margin
            "lambda": 1e-2,     # 量化损失权重
        },
        
        # 训练参数
        "training_params": {
            "epoch": 180,
            "save_epoch_start": 50,
            "step_num": 10,
            "log_interval": 5,  # 参考DSH每5个epoch验证一次
        },
        
        # 评估参数
        "eval_params": {
            "topK": dataset_config["topK"],
            "topK_mAP": dataset_config["topK_mAP"],
            "r": dataset_config["r"],
        },
        
        # 路径参数
        "path_params": {
            "save_path": dataset_config["save_path"],
            "log_dir": dataset_config["log_dir"],
        },
        
        # 硬件参数
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        
        # hash bit数量
        "bit_list": [16],  # 参考DSH的顺序
    }
    
    # 确保目录存在
    os.makedirs(config["path_params"]["save_path"], exist_ok=True)
    os.makedirs(config["path_params"]["log_dir"], exist_ok=True)
    
    # 展平配置
    flat_config = {}
    for category, params in config.items():
        if isinstance(params, dict) and category != "optimizer":
            for key, value in params.items():
                flat_config[key] = value
        else:
            flat_config[category] = params
    
    return flat_config


class DTSHLoss(torch.nn.Module):
    """DTSH损失函数实现"""
    def __init__(self, config):
        super(DTSHLoss, self).__init__()
        self.alpha = config["alpha"]  # triplet margin
        self.lambda_param = config["lambda"]  # 量化损失权重
        
    def forward(self, u, y, config):
        """
        计算DTSH损失
        
        Args:
            u: 连续哈希码 (batch_size, hash_bits)
            y: 标签 (batch_size, num_classes)
            config: 配置参数
            
        Returns:
            总损失
        """
        batch_size = u.shape[0]
        
        # 计算内积
        inner_product = u @ u.t()
        
        # 计算相似性矩阵
        s = (y @ y.t() > 0).float()
        
        # triplet损失
        count = 0
        loss1 = 0
        
        for row in range(batch_size):
            # 找到正样本和负样本
            positive_mask = s[row] == 1
            negative_mask = s[row] == 0
            
            # 如果既有正样本又有负样本
            if positive_mask.sum() > 0 and negative_mask.sum() > 0:
                count += 1
                
                # 正样本的内积
                theta_positive = inner_product[row][positive_mask]
                # 负样本的内积
                theta_negative = inner_product[row][negative_mask]
                
                # 计算triplet损失: max(0, alpha - theta_pos + theta_neg)
                # 这里使用broadcast计算所有可能的triplet组合
                triplet_loss = (theta_positive.unsqueeze(1) - theta_negative.unsqueeze(0) - self.alpha)
                triplet_loss = triplet_loss.clamp(min=-100, max=50)  # 防止数值溢出
                
                # 使用log-sum-exp trick计算softmax交叉熵
                loss1 += -(triplet_loss - torch.log(1 + torch.exp(triplet_loss))).mean()
        
        if count != 0:
            loss1 = loss1 / count
        else:
            loss1 = torch.tensor(0.0, device=u.device)
        
        # 量化损失: ||u - sign(u)||^2
        loss2 = self.lambda_param * (u - u.sign()).pow(2).mean()
        
        return loss1 + loss2


def train_val(config, bit):
    """训练和验证函数"""
    log_dir = os.path.join(config["log_dir"], f"{config['dataset']}_{bit}bits")
    writer = SummaryWriter(log_dir=log_dir)
    
    device = config["device"]
    train_loader, test_loader, valid_loader, num_train, num_test, num_valid = get_data(config)
    config["num_train"] = num_train
    config["num_test"] = num_test
    config["num_valid"] = num_valid
    config["batch_num"] = num_train // config["batch_size"] + 1
    
    print(f"网络: {config['info']}, 数据集: {config['dataset']}, Hash位数: {bit}")
    print(f"训练样本: {num_train}, 验证样本: {num_valid}, 测试样本: {num_test}")
    
    # 初始化网络
    net = config["net"](config["info"], config["hidden_dim"], bit, config["beta"]).to(device)
    
    # AlexNet优化器参数设置
    parameter_list = [
        {"params": net.feature_layers.parameters(), "lr": 1},
        {"params": net.hash_layers.parameters(), "lr": 10}
    ]
    
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, **(config["optimizer"]["optim_params"]))
    
    # 学习率调度器设置
    param_lr = []
    layers = ["feature_layers", "hash_layers"]
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    
    config["Network_init_lr"] = dict(zip(layers, param_lr))
    config["max_iter"] = config["epoch"] * config["batch_num"]
    optimizer_config["lr_param"]["step"] = math.ceil(config["max_iter"] / config["step_num"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[config["optimizer"]["lr_type"]]
    
    print(f"学习率调度策略: {lr_scheduler}")
    print(f"初始学习率: {config['Network_init_lr']}")
    
    print(config)
    
    criterion = DTSHLoss(config)
    
    # 训练记录
    Best_mAP = 0.0
    Best_AP_topK = 0.0
    train_losses = []
    training_times = []
    # validation results
    val_maps = []
    val_ap_topKs = []
    val_ap_rs = []
    iter_num = 0
    decay_times = 0
    # test results
    tst_mAP = 0.0
    tst_AP_r = 0.0
    tst_AP_topK = 0.0
    tst_results = []
    
    for epoch in range(config["epoch"]):
        start = time.time()
        net.train()
        train_loss = 0
        
        print(f"Epoch {epoch + 1}/{config['epoch']} - 训练阶段")
        for image, label, ind in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
            image = image.to(device)
            label = label.to(device)
            
            # 学习率调度
            decay_times, optimizer = lr_scheduler(param_lr, optimizer, iter_num, **schedule_param)
            
            optimizer.zero_grad()
            u = net(image)
            loss = criterion(u, label.float(), config)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            iter_num += 1
        
        train_loss = train_loss / num_train
        train_losses.append(train_loss)
        end = time.time()
        duration = end - start
        training_times.append(duration)
        
        # 每5个epoch进行一次验证
        if (epoch + 1) % config["log_interval"] == 0 or epoch == 0:
            print(f"Epoch {epoch + 1} - 验证阶段")
            net.eval()
            
            # 计算验证集结果
            val_binary, val_label = compute_result(valid_loader, net, device=device)
            trn_binary, trn_label = compute_result(train_loader, net, device=device)
            
            if config["dataset"] == "cifar-10": 
        
                val_mAP = hash_ranking_map(
                    trn_binary.cpu().numpy(), 
                    trn_label.cpu().numpy(), 
                    val_binary.cpu().numpy(), 
                    val_label.cpu().numpy()
                )
            else:
                val_mAP = hash_ranking_map_topk(
                    trn_binary.cpu().numpy(), 
                    trn_label.cpu().numpy(), 
                    val_binary.cpu().numpy(), 
                    val_label.cpu().numpy(), 
                    topk=config["topK_mAP"]
                )
            val_AP_topK, _ = get_precision_recall_topK(
                trn_binary.cpu().numpy(), 
                trn_label.cpu().numpy(), 
                val_binary.cpu().numpy(), 
                val_label.cpu().numpy(), 
                topk=config["topK"]
            )
            
            val_AP_r, _ = get_precision_recall_within_hamming_radius(
                trn_binary.cpu().numpy(), 
                trn_label.cpu().numpy(), 
                val_binary.cpu().numpy(), 
                val_label.cpu().numpy(), 
                r=config["r"]
            )
            
            # 记录验证结果
            val_maps.append(val_mAP)
            val_ap_topKs.append(val_AP_topK)
            val_ap_rs.append(val_AP_r)
            
            # TensorBoard日志
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('mAP/val', val_mAP, epoch)
            writer.add_scalar('AP_topK/val', val_AP_topK, epoch)
            writer.add_scalar('AP_r/val', val_AP_r, epoch)
            
            print(f"验证结果 - mAP: {val_mAP:.4f}, AP@topK: {val_AP_topK:.4f}, AP@r: {val_AP_r:.4f}")
            
            # 如果验证性能提升，在测试集上评估
            if val_mAP > Best_mAP:
                print(f"验证mAP提升 {Best_mAP:.4f} -> {val_mAP:.4f}, 开始测试集评估...")
                Best_mAP = val_mAP
                
                # 测试集评估
                tst_binary, tst_label = compute_result(test_loader, net, device=device)
                if config["dataset"] == "nus-wide": 
                    tst_mAP = hash_ranking_map_topk(
                        trn_binary.cpu().numpy(), 
                        trn_label.cpu().numpy(), 
                        tst_binary.cpu().numpy(), 
                        tst_label.cpu().numpy(), 
                        topk=config["topK_mAP"]
                    )
                    
                else:
                    tst_mAP = hash_ranking_map(
                        trn_binary.cpu().numpy(), 
                        trn_label.cpu().numpy(), 
                        tst_binary.cpu().numpy(), 
                        tst_label.cpu().numpy()
                    )
                    
                tst_AP_topK, _ = get_precision_recall_topK(
                    trn_binary.cpu().numpy(), 
                    trn_label.cpu().numpy(), 
                    tst_binary.cpu().numpy(), 
                    tst_label.cpu().numpy(), 
                    topk=config["topK"]
                )
                tst_AP_r, _ = get_precision_recall_within_hamming_radius(
                    trn_binary.cpu().numpy(), 
                    trn_label.cpu().numpy(), 
                    tst_binary.cpu().numpy(), 
                    tst_label.cpu().numpy(), 
                    r=config["r"]
                )
                
                tst_results = [tst_mAP, tst_AP_topK, tst_AP_r]
                
                # TensorBoard测试日志
                writer.add_scalar('mAP/test', tst_mAP, epoch)
                writer.add_scalar('AP_topK/test', tst_AP_topK, epoch)
                writer.add_scalar('AP_r/test', tst_AP_r, epoch)
                
                print(f"测试结果 - mAP: {tst_mAP:.4f}, AP@topK: {tst_AP_topK:.4f}, AP@r: {tst_AP_r:.4f}")
                
                # 保存测试结果到文件
                results_path = os.path.join(config["save_path"], f"{config['dataset']}_{bit}bits")
                os.makedirs(results_path, exist_ok=True)
                results_file = os.path.join(results_path, "test_results.txt")
                
                if not os.path.exists(results_file):
                    with open(results_file, 'w') as f:
                        f.write("epoch tst_mAP tst_AP_topK tst_AP_r\n")

                with open(results_file, 'a') as f:
                    f.write(f"{epoch+1} {tst_mAP:.6f} {tst_AP_topK:.6f} {tst_AP_r:.6f}\n")

                # 保存最佳模型
                if epoch >= config["save_epoch_start"]:
                    best_model_path = os.path.join(config["save_path"], f'{config["dataset"]}_{bit}bits_mAP_{tst_mAP:.4f}')
                    os.makedirs(best_model_path, exist_ok=True)
                    
                    np.save(os.path.join(best_model_path, "val_label.npy"), val_label.numpy())
                    np.save(os.path.join(best_model_path, "val_binary.npy"), val_binary.numpy())
                    np.save(os.path.join(best_model_path, "trn_binary.npy"), trn_binary.numpy())
                    np.save(os.path.join(best_model_path, "trn_label.npy"), trn_label.numpy())
                    np.save(os.path.join(best_model_path, "tst_binary.npy"), tst_binary.numpy())
                    np.save(os.path.join(best_model_path, "tst_label.npy"), tst_label.numpy())
                    np.save(os.path.join(best_model_path, "tst_results.npy"), tst_results)
                    torch.save(net.state_dict(), os.path.join(best_model_path, "model.pt"))
                    print(f"模型已保存到: {best_model_path}")
                    
            if val_AP_topK > Best_AP_topK:
                Best_AP_topK = val_AP_topK
        else:
            # 非验证epoch只记录损失
            val_maps.append(val_maps[-1] if val_maps else 0.0)
            val_ap_topKs.append(val_ap_topKs[-1] if val_ap_topKs else 0.0)
            val_ap_rs.append(val_ap_rs[-1] if val_ap_rs else 0.0)

        # 每个epoch都输出简要信息
        print(f"Epoch {epoch + 1}/{config['epoch']}: "
              f"loss={train_loss:.4f}, "
              f"time={duration:.1f}s, "
              f"best_val_mAP={Best_mAP:.4f}, "
              f"test_mAP={tst_mAP:.4f}")
    
    # 保存训练历史
    save_path = os.path.join(config["save_path"], f"{config['dataset']}_{bit}bits")
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, "val_mAP.npy"), val_maps)
    np.save(os.path.join(save_path, "val_AP_topK.npy"), val_ap_topKs)
    np.save(os.path.join(save_path, "val_AP_r.npy"), val_ap_rs)
    np.save(os.path.join(save_path, "train_loss.npy"), train_losses)
    np.save(os.path.join(save_path, "time.npy"), training_times)
    
    writer.close()
    
    print(f"\n{'='*60}")
    print(f"DTSH {bit}-bit 训练完成!")
    print(f"最佳验证mAP: {Best_mAP:.4f}")
    print(f"最终测试mAP: {tst_mAP:.4f}")
    print(f"最终测试AP@topK: {tst_AP_topK:.4f}")
    print(f"最终测试AP@r: {tst_AP_r:.4f}")
    print(f"{'='*60}\n")
    
    return tst_mAP, tst_AP_topK, tst_AP_r


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default= 'nus-wide', 
                        choices=['nus-wide','cifar-10'],
                        help='Dataset to use: cifar-10 or nus-wide')
    args = parser.parse_args()
    
    config = get_config(args.dataset)
    
    margin = 0
    
    print(f"Using dataset: {args.dataset}")
    
    for bit in config["bit_list"]:
        margin = bit / 2
        config["alpha"] = 10
        train_val(config, bit)