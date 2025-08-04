import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from tqdm import tqdm
import math 
import json 
import random
from scipy.linalg import hadamard
from scipy.special import comb
from torch.utils.tensorboard import SummaryWriter

from network import AlexNet as OriginalAlexNet
from sgd_tools import *

from MDSHC_network import MoCoAlexNet
from MDSHC_network_resnet import MoCoResNet50

from MDSHC_loss import CenterHashLoss
from lr_schedule import adjust_learning_rate_multistep

# 设置多进程共享策略
torch.multiprocessing.set_sharing_strategy('file_system')

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

def get_center_hash_config(network_choice="alexnet"):
    """获取中心哈希方法的默认配置"""
    # 网络配置选项
    network_configs = {
        "alexnet": {
            "net_class": MoCoAlexNet,
            "info": "MDSHC_AlexNet",
            "feature_lr": 0.001,
            "hash_lr": 0.01,
        },
        "resnet50": {
            "net_class": MoCoResNet50,
            "info": "MDSHC_ResNet50", 
            "feature_lr": 0.0001,  # ResNet需要更小的学习率
            "hash_lr": 0.001,
        }
    }
    
    network_config = network_configs[network_choice]
    
    config = {
        # 中心哈希损失参数
        "loss_params": {
            "beta": 1.0,            # 对比损失权重
            "lambda": 0.0001,       # 量化损失权重
            "epoch_change": 3,      # 开始使用对比损失的轮次
            "cos_scale": 10.0,      # 余弦相似度缩放因子
            "moco_temperature": 0.1 # MoCo对比损失温度参数
        },
        
        # 网络配置
        "net_class": network_config["net_class"],
        "info": network_config["info"],
        
        # 优化器设置 (根据原论文)
        "optimizer_type": "SGD",     
        "optimizer": {
            "feature_lr": 0.001,    # 特征提取层学习率
            "hash_lr": 0.01,        # 哈希层学习率
            "momentum": 0.905,        # 动量系数
            "weight_decay": 0.0005, # 权重衰减
            "nesterov": True,       # 使用 Nesterov 动量
        },
        
        # 学习率调度
        "lr_scheduler_type": "multistep",
        "lr_scheduler_params": {
            "milestones": "120,160", # 学习率降低点
            "gamma": 0.7            # 学习率降低系数
        },

        # 数据与模型结构
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 128,          # 原论文批量大小
        "hidden_dim": 4096,         # 隐藏层维度

        # 数据集设置
        "dataset": "cifar-10",      # 数据集名称
        "n_class": 10,              # 类别数量
        
        # "dataset": "nus-wide",  
        # "n_class": 21,              # 类别数量

        # 训练与评估
        "epoch": 160,               # 原论文总训练轮数
        "topK": 1000,               # AP@K评估参数
        "topK_mAP" : 5000,          # 针对nus-wide的mAP范围
        "r": 2,                     # Hamming Radius评估参数
        "eval_interval": 5,         # 评估间隔

        # 保存与设备设置
       "save_path": "../autodl-tmp/save/ResNet50/CIFAR10", 
        "log_dir": "../tf-logs/MDSHC/ResNet50/CIFAR10", 
        "log_interval": 5,          # 日志记录间隔
        "save_epoch_start": 70,     # 开始保存模型的轮次
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "bit_list": [64],           # 哈希码位数
#         16,32,48, 
        "order_seed": 80,           # 哈希中心顺序随机化种子
        "seed": 2024              # 随机种子
    }
    return config

def get_margin(bit, n_class):
    """计算基于Gilbert-Varshamov边界的最小距离"""
    L = bit
    right = (2 ** L) / n_class
    d_min = 0
    d_max = 0
    
    # 计算d_min
    for j in range(2 * L + 4):
        dim = j
        sum_1 = 0
        sum_2 = 0
        for i in range((dim - 1) // 2 + 1):
            sum_1 += comb(L, i)
        for i in range((dim) // 2 + 1):
            sum_2 += comb(L, i)
        if sum_1 <= right and sum_2 > right:
            d_min = dim
    
    # 计算d_max
    for i in range(2 * L + 4):
        dim = i
        sum_1 = 0
        sum_2 = 0
        for j in range(dim):
            sum_1 += comb(L, j)
        for j in range(dim - 1):
            sum_2 += comb(L, j)
        if sum_1 >= right and sum_2 < right:
            d_max = dim
            break
    
    return d_max, d_min

def CSQ_init(n_class, bit):
    """初始化哈希中心（使用CSQ方法）"""
    h_k = hadamard(bit)
    h_2k = np.concatenate((h_k, -h_k), 0)
    hash_center = h_2k[:n_class]

    # 如果哈希中心数量不足，补充随机生成的中心
    if h_2k.shape[0] < n_class:
        hash_center = np.resize(hash_center, (n_class, bit))
        for k in range(5):
            for index in range(h_2k.shape[0], n_class):
                ones = np.ones(bit)
                ones[random.sample(list(range(bit)), bit // 2)] = -1
                hash_center[index] = ones
            c = []
            for i in range(n_class):
                for j in range(i, n_class):
                    c.append(sum(hash_center[i] != hash_center[j]))
            c = np.array(c)
            if c.min() > bit / 4 and c.mean() >= bit / 2:
                break
    return hash_center

def init_hash(n_class, bit):
    """随机初始化哈希中心"""
    hash_centers = -1 + 2 * np.random.random((n_class, bit))
    hash_centers = np.sign(hash_centers)
    mean_dist, min_dist, var_dist, max_dist = evaluate_centers(hash_centers)
    print(f"Random init: mean={mean_dist:.2f}, min={min_dist}, var={var_dist:.2f}, max={max_dist}")
    return hash_centers

def compute_result(dataloader, net, device):
    """计算数据集上的哈希码和标签"""
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader, desc="计算哈希码", leave=False):
        clses.append(cls)
        # MoCoAlexNet 返回两个输出，只取第一个
        bs.append(net(img.to(device))[0].data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

def evaluate_centers(H):
    """评估哈希中心之间的距离分布"""
    dist = []
    for i in range(H.shape[0]):
        for j in range(i + 1, H.shape[0]):
            # 计算汉明距离
            TF = np.sum(H[i] != H[j])
            dist.append(TF)
    dist = np.array(dist)
    mean_dist = dist.mean() if len(dist) > 0 else 0
    min_dist = dist.min() if len(dist) > 0 else 0
    var_dist = dist.var() if len(dist) > 0 else 0
    max_dist = dist.max() if len(dist) > 0 else 0
    
    print(f"哈希中心评估: mean={mean_dist:.2f}, min={min_dist}, max={max_dist}, var={var_dist:.2f}")
    return mean_dist, min_dist, var_dist, max_dist

def train_val_center_hash(config, bit):
    """训练和验证中心哈希模型"""
    # 日志和设备设置
    log_dir_suffix = f"{config['dataset']}_{bit}bits"
    log_dir = os.path.join(config["log_dir"], log_dir_suffix)
    writer = SummaryWriter(log_dir=log_dir)
    
    device = config["device"]
    print(f"使用设备: {device}")

    # 数据加载
    print(f"正在加载数据集: {config['dataset']}")
    train_loader, test_loader, valid_loader, num_train, num_test, num_valid = get_data(config)
    config["num_train"] = num_train
    config["num_test"] = num_test
    config["num_valid"] = num_valid
    config["batch_num"] = num_train // config["batch_size"] + (1 if num_train % config["batch_size"] != 0 else 0)
    
    print(f"  训练集大小: {num_train}")
    print(f"  测试集大小: {num_test}")
    print(f"  验证集大小: {num_valid}")
    
    # 加载或生成哈希中心
    center_dir = f"./centerswithoutVar/{config['dataset']}"
    os.makedirs(center_dir, exist_ok=True)
    center_path = f"{center_dir}/centers_{config['n_class']}_{bit}.npy"
    config["center_path"] = center_path
    
    if os.path.exists(center_path):
        print(f"加载预计算的哈希中心: {center_path}")
    else:
        print(f"未找到哈希中心文件，生成新的哈希中心...")
        # 计算最小距离
        d_max, d_min = get_margin(bit, config['n_class'])
        print(f"计算的最小距离: d_max={d_max}, d_min={d_min}")
        
        # 初始化哈希中心
        initWithCSQ = True
        if bit == 48:
            initWithCSQ = False  # 48 位使用随机初始化
        if initWithCSQ:
            hash_centers = CSQ_init(config['n_class'], bit)
        else:
            hash_centers = init_hash(config['n_class'], bit)
        
        # 评估并保存哈希中心
        mean_dist, min_dist, var_dist, max_dist = evaluate_centers(hash_centers)
        os.makedirs(os.path.dirname(center_path), exist_ok=True)
        np.save(center_path, hash_centers)
        print(f"哈希中心已保存到: {center_path}")
    
    # 设置随机类别顺序
    l = list(range(config['n_class']))
    random.seed(config['order_seed'])
    random.shuffle(l)
    
   # 模型实例化 - 使用配置中的网络类
    print("初始化模型...")
    net = config["net_class"](
        hidden_dim=config["hidden_dim"], 
        hash_bits=bit,
        pretrained=True
    ).to(device)
    
    # 损失函数实例化
    criterion = CenterHashLoss(config, bit, device)
    
    # 优化器设置
    feature_lr = config["optimizer"]["feature_lr"]
    hash_lr = config["optimizer"]["hash_lr"]
    
    # 分离参数组 - 特征提取部分和哈希部分

#     backbone_params = net.encoder_q.feature_extractor.parameters()
#     hash_params = list(net.encoder_q.hash_head.parameters()) + list(net.encoder_k.parameters())

#     parameter_list = [
#         {"params": backbone_params, "lr": feature_lr},
#         {"params": hash_params, "lr": hash_lr}
#     ]
    
    # 分离参数组 - 特征提取部分和哈希部分
# 根据网络类型使用正确的属性名
    if 'ResNet50' in config.get("info", ""):
        # ResNet50 使用 'feature_layers' 和 'hash_layers'
        print("为 ResNet50 分离参数: feature_layers, hash_layers")
        backbone_params = net.encoder_q.feature_layers.parameters()
        hash_params = net.encoder_q.hash_layers.parameters()
    else:
        # AlexNet 使用 'feature_extractor' 和 'hash_head'
        print("为 AlexNet 分离参数: feature_extractor, hash_head")
        backbone_params = net.encoder_q.feature_extractor.parameters()
        hash_params = net.encoder_q.hash_head.parameters()

    # MoCo中的encoder_k参数通过动量更新，不应加入优化器
    parameter_list = [
        {"params": backbone_params, "lr": feature_lr},
        {"params": hash_params, "lr": hash_lr}
    ]
    
    optimizer = optim.SGD(
        parameter_list,
        momentum=config["optimizer"]["momentum"],
        weight_decay=config["optimizer"]["weight_decay"],
        nesterov=config["optimizer"]["nesterov"]
    )
    
    print(f"  优化器: {config['optimizer_type']}")
    print(f"  初始特征层学习率: {feature_lr}, 初始哈希层学习率: {hash_lr}")
    
   # 训练过程变量初始化
    Best_mAP = 0.0
    train_loss_list = []
    time_list = []
    map_list = []
    ap_topK_list = []
    ap_r_list = []
    
    # 开始训练循环
    print(f"\n--- 开始训练 {bit} 位哈希码 ---")
    training_start_time = time.time()
    
    for epoch in range(config["epoch"]):
        print(f"\n--- Epoch {epoch + 1}/{config['epoch']} ---")
        epoch_start_time = time.time()
        net.train()
        train_loss = 0
        train_center_loss = 0
        train_pair_loss = 0
        train_quan_loss = 0
        
        # 调整学习率
        current_lr = adjust_learning_rate_multistep(optimizer, epoch, config)
        
        # 训练一个epoch
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1} 训练", leave=True)
        for batch_idx, (image, label, ind) in enumerate(batch_iterator):
            image = image.to(device)
            label = label.to(device).float() 
            ind = ind.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            u1, u2 = net(image)
            
            # 计算损失
            loss, loss_dict = criterion(u1, u2, label, ind, epoch)
            
            # 反向传播和优化
            loss.backward()
            # 移除梯度裁剪 (根据你的要求)
            optimizer.step()
            
            # 累加损失
            train_loss += loss_dict["total_loss"]
            train_center_loss += loss_dict["center_loss"]
            train_pair_loss += loss_dict["pair_loss"]
            train_quan_loss += loss_dict["quan_loss"]
            
            # 更新进度条
            batch_iterator.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'center': f"{loss_dict['center_loss']:.4f}",
                'pair': f"{loss_dict['pair_loss']:.4f}" if epoch >= config['loss_params']['epoch_change'] else "0.0000"
            })
        
        # 计算平均损失
        avg_train_loss = train_loss / len(train_loader)
        avg_center_loss = train_center_loss / len(train_loader)
        avg_pair_loss = train_pair_loss / len(train_loader)
        avg_quan_loss = train_quan_loss / len(train_loader)
        
        train_loss_list.append(avg_train_loss)
        
        # 记录本轮用时
        epoch_end_time = time.time()
        duration = epoch_end_time - epoch_start_time
        time_list.append(duration)
        
        # 记录训练日志
        if writer:
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Loss/center', avg_center_loss, epoch)
            writer.add_scalar('Loss/pair', avg_pair_loss, epoch)
            writer.add_scalar('Loss/quan', avg_quan_loss, epoch)
            writer.add_scalar('LearningRate/feature', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('LearningRate/hash', optimizer.param_groups[1]['lr'], epoch)
        
        # 验证
        if (epoch + 1) % config["eval_interval"] == 0:
            print("--- 验证中 ---")
            val_binary, val_label = compute_result(valid_loader, net, device)
            print(f"  验证集哈希码计算完成，形状: {val_binary.shape}")
            
            trn_binary, trn_label = compute_result(train_loader, net, device)
            print(f"  训练集哈希码计算完成，形状: {trn_binary.shape}")
            
            # 计算评估指标
            if config["dataset"] == "nus-wide":
                mAP = hash_ranking_map_topk(trn_binary.cpu().numpy(), trn_label.cpu().numpy(), val_binary.cpu().numpy(), val_label.cpu().numpy(), topk=config["topK_mAP"])
            else:
                mAP = hash_ranking_map(trn_binary.cpu().numpy(), trn_label.cpu().numpy(), 
                                   val_binary.cpu().numpy(), val_label.cpu().numpy())
            
            AP_topK, _ = get_precision_recall_topK(
                trn_binary.cpu().numpy(), trn_label.cpu().numpy(),
                val_binary.cpu().numpy(), val_label.cpu().numpy(),
                topk=config["topK"]
            )
            
            AP_r, _  = get_precision_recall_within_hamming_radius(
                trn_binary.cpu().numpy(), trn_label.cpu().numpy(),
                val_binary.cpu().numpy(), val_label.cpu().numpy(),
                r=config["r"]
            )
            
            print(f"  验证集 mAP: {mAP:.4f}, AP@{config['topK']}: {AP_topK:.4f}, AP@R<={config['r']}: {AP_r:.4f}")
            
            # 记录验证指标
            if writer:
                writer.add_scalar('mAP/val', mAP, epoch)
                writer.add_scalar('AP_topK/val', AP_topK, epoch)
                writer.add_scalar('AP_r/val', AP_r, epoch)
            
            # 如果是最佳模型则测试并保存
            if mAP > Best_mAP:
                Best_mAP = mAP
                print(f"*** 新的最佳验证集 mAP: {Best_mAP:.4f} at Epoch {epoch + 1} ***")
                
                # 在测试集上评估
                print("--- 使用最佳模型进行测试 ---")
                tst_binary, tst_label = compute_result(test_loader, net, device)
                print(f"  测试集哈希码计算完成，形状: {tst_binary.shape}")
                if config["dataset"] == "nus-wide":
                    tst_mAP = hash_ranking_map_topk(
                    trn_binary.cpu().numpy(), trn_label.cpu().numpy(),
                    tst_binary.cpu().numpy(), tst_label.cpu().numpy(),
                    topk=config["topK_mAP"]
                )
                else:
                    tst_mAP = hash_ranking_map(
                    trn_binary.cpu().numpy(), trn_label.cpu().numpy(),
                    tst_binary.cpu().numpy(), tst_label.cpu().numpy()
                )
                
                tst_AP_topK, _ = get_precision_recall_topK(
                    trn_binary.cpu().numpy(), trn_label.cpu().numpy(),
                    tst_binary.cpu().numpy(), tst_label.cpu().numpy(),
                    topk=config["topK"]
                )
                
                tst_AP_r, _ = get_precision_recall_within_hamming_radius(
                    trn_binary.cpu().numpy(), trn_label.cpu().numpy(),
                    tst_binary.cpu().numpy(), tst_label.cpu().numpy(),
                    r=config["r"]
                )
                
                # 收集测试结果
                tst_results = [tst_mAP, tst_AP_topK, tst_AP_r]
                
                print(f"  测试集结果: mAP={tst_mAP:.4f}, AP@{config['topK']}={tst_AP_topK:.4f}, AP@R<={config['r']}={tst_AP_r:.4f}")
                
                # 记录测试指标
                if writer:
                    writer.add_scalar('mAP/test', tst_mAP, epoch)
                    writer.add_scalar('AP_topK/test', tst_AP_topK, epoch)
                    writer.add_scalar('AP_r/test', tst_AP_r, epoch)
                
                # 保存测试结果
                results_path = os.path.join(config["save_path"], f"test_results/{config['dataset']}_{bit}bits")
                os.makedirs(results_path, exist_ok=True)
                results_file = os.path.join(results_path, "test_summary.txt")
                
                if not os.path.exists(results_file):
                    with open(results_file, 'w') as f:
                        f.write("Epoch tst_mAP tst_AP_topK tst_AP_r\n")
                
                with open(results_file, 'a') as f:
                    f.write(f"{epoch+1} {tst_mAP:.6f} {tst_AP_topK:.6f} {tst_AP_r:.6f}\n")
                
                # 保存最佳模型
                if epoch >= config["save_epoch_start"]:
                    best_model_path = os.path.join(config["save_path"], f'models/{config["dataset"]}_{bit}bits_best_e{epoch+1}_mAP{Best_mAP:.4f}')
                    os.makedirs(best_model_path, exist_ok=True)
                    
                    # 保存模型权重
                    torch.save(net.state_dict(), os.path.join(best_model_path, "model.pt"))
                    
                    # 保存哈希码和标签
                    np.save(os.path.join(best_model_path, "val_binary.npy"), val_binary.cpu().numpy())
                    np.save(os.path.join(best_model_path, "val_label.npy"), val_label.cpu().numpy())
                    np.save(os.path.join(best_model_path, "trn_binary.npy"), trn_binary.cpu().numpy())
                    np.save(os.path.join(best_model_path, "trn_label.npy"), trn_label.cpu().numpy())
                    np.save(os.path.join(best_model_path, "tst_binary.npy"), tst_binary.cpu().numpy())
                    np.save(os.path.join(best_model_path, "tst_label.npy"), tst_label.cpu().numpy())
                    np.save(os.path.join(best_model_path, "tst_results.npy"), np.array(tst_results))
                    
                    print(f"最佳模型和结果已保存到: {best_model_path}")
            
            # 记录当前轮次的验证指标
            map_list.append(mAP)
            ap_topK_list.append(AP_topK)
            ap_r_list.append(AP_r)
            
            # 打印本轮总结
            print(f"Epoch {epoch + 1}/{config['epoch']} 总结: "
                  f"数据集: {config['dataset']}, 哈希位数: {bit}, "
                  f"验证集 AP@{config['topK']}: {AP_topK:.4f}, AP@R<={config['r']}: {AP_r:.4f}, mAP: {mAP:.4f}, "
                  f"最佳验证集 mAP: {Best_mAP:.4f}, "
                  f"用时: {duration:.2f}s, 训练损失: {avg_train_loss:.4f}")
    
    # 训练完成，保存最终结果
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    print(f"\n--- {bit} 位哈希码训练完成 ---")
    print(f"总训练时间: {total_training_time / 3600:.2f} 小时")
    print(f"最终最佳验证集 mAP: {Best_mAP:.6f}")
    
    # 保存训练历史
    final_results_path = os.path.join(config["save_path"], f"final_results/{config['dataset']}_{bit}bits")
    os.makedirs(final_results_path, exist_ok=True)
    
    np.save(os.path.join(final_results_path, "mAP_history.npy"), np.array(map_list))
    np.save(os.path.join(final_results_path, "AP_topK_history.npy"), np.array(ap_topK_list))
    np.save(os.path.join(final_results_path, "AP_r_history.npy"), np.array(ap_r_list))
    np.save(os.path.join(final_results_path, "train_loss_history.npy"), np.array(train_loss_list))
    np.save(os.path.join(final_results_path, "epoch_time_history.npy"), np.array(time_list))
    
    # 保存最终模型
    torch.save(net.state_dict(), os.path.join(final_results_path, "model_final.pt"))
    
    # 保存配置参数
    config_to_save = config.copy()
    config_to_save['device'] = str(config['device'])
    with open(os.path.join(final_results_path, "config.json"), 'w') as f:
        json.dump(config_to_save, f, indent=4)
    
    print(f"最终结果和配置已保存到: {final_results_path}")
    
    if writer:
        writer.close()
    
    return Best_mAP

# 4. 修改main函数，添加命令行参数
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='alexnet',
                        choices=['alexnet', 'resnet50'],
                        help='Network to use: alexnet or resnet50')
    parser.add_argument('--dataset', type=str, default='cifar-10',
                        choices=['cifar-10', 'nus-wide'],
                        help='Dataset to use')
    args = parser.parse_args()
    
    # 获取配置
    config = get_center_hash_config(args.network)
    
    # 根据数据集调整配置
    if args.dataset == "nus-wide":
        config["dataset"] = "nus-wide"
        config["n_class"] = 21
        config["topK_mAP"] = 5000
    
    # 设置随机种子
    setup_seed(config['seed'])
    
    print(f"MDSHC: {args.network} on {args.dataset}")
    
    # 对每个哈希位数进行训练
    for bit in config['bit_list']:
        print(f"\n{'='*20} 开始训练 {bit} 位哈希码 {'='*20}\n")
        best_map = train_val_center_hash(config, bit)
        print(f"\n{'='*20} {bit} 位哈希码训练完成, 最佳 mAP: {best_map:.6f} {'='*20}\n")