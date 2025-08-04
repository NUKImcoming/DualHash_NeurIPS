# orthohash_baseline_runner.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from tqdm import tqdm
import math 
import json 
from torch.utils.tensorboard import SummaryWriter # 用于日志记录

# --- 导入你的和 OrthoHash 相关的模块 ---
from network import AlexNet as OriginalAlexNet # 你的原始 AlexNet (用于提取 backbone)
from sgd_tools import *
from Orthohash_network import OrthoHashAlexNetBaseline, CosSim # OrthoHash 网络
from orthohash_loss import OrthoHashLoss # 简化的 OrthoHash 损失
from codebook_utils import generate_codebook # Codebook 生成器
from lr_schedule import adjust_learning_rate_multistep

torch.multiprocessing.set_sharing_strategy('file_system')

# --- 设置随机种子 ---
seed = 2024 
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = False # 通常为了速度设为 False
    torch.backends.cudnn.benchmark = True
print(f"Seed set to {seed}")

# --- 配置函数 (基于 ablation_config.py 风格修改) ---
def get_orthohash_config():
    # !!! 重要: 确保这里的参数与你的 ablation_config.py 中的 SGD 部分一致 !!!
    config = {
        # OrthoHash Loss 相关参数
        "ortho_ce": 1.0,             
        "ortho_s": 8.0,              # OrthoHash 常用值
        "ortho_m": 0.2,              # OrthoHash 常用值
        "ortho_m_type": "cos",       

        # Codebook 相关
        "codebook_method": 'B',      # 'N', 'B', 'O'

        # 优化器 (保持 SGD + Nesterov)
        "optimizer_type": "SGD",     # 固定为 SGD
        "optimizer": {
            "feature_lr": 0.0001,     # 
            "hash_lr": 0.001,         #
            "momentum": 0.905,        #
            "weight_decay": 0.0005,  # 
            "nesterov": True,        # 
        },
        # 学习率
        "lr_scheduler_type": "multistep", # 使用 multistep
        "lr_scheduler_params": {
            "milestones": "120,160", # <<< 设置衰减点
            "gamma": 0.7            # <<< 设置衰减率
        },

        # 数据与模型结构
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 128,
        "hidden_dim": 4096,         # 你的 AlexNet Backbone 输出维度

        # 数据集设置 (修改为NUS-WIDE)
        "dataset": "nus-wide",     # 从"cifar-10"改为"nus-wide"
        "num_classes": 21,          # NUS-WIDE有21个类别

        # 训练与评估
        "epoch": 200,               # !!! 与你的 ablation_config epoch 保持一致 !!!
        "topK": 1000,               # !!! 与你的 ablation_config topK 保持一致 !!!
        "topK_mAP": 5000,           # 添加与DHN相同的参数
        "r": 2,                     # !!! 与你的 ablation_config r 保持一致 !!!
        "eval_interval": 5,        # 每隔多少 epoch 评估一次

        # 保存与设备
        "save_path": "/root/autodl-tmp/save/OrthoHash/NUS-WIDE", # 修改路径
        "log_dir": "/root/tf-logs/OrthoHash/NUS-WIDE", # 修改日志路径
        "log_interval": 5,         
        "save_epoch_start": 70,     
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "bit_list": [16, 32, 48, 64], # 
        "info": "OrthoHash" 
    }
    return config

# --- 优化器创建函数 (只保留 SGD) ---
def create_optimizer(config, parameter_list):
    """根据配置创建 SGD 优化器"""
    if config["optimizer_type"] == "SGD":
        print("Creating SGD optimizer...")
        optimizer_params = config["optimizer"]
        return torch.optim.SGD(
            parameter_list,
            momentum=optimizer_params["momentum"],
            weight_decay=optimizer_params["weight_decay"],
            nesterov=optimizer_params["nesterov"]
        )
    else:
        raise ValueError(f"Unsupported optimizer type for this script: {config['optimizer_type']}. Only SGD is expected.")

# --- 适配 OrthoHash 输出的 compute_result ---
def compute_result_orthohash(dataloader, net, device):
    """Adapted compute_result for OrthoHash model output (u, v)."""
    bs, clses = [], []
    net.eval() # 切换到评估模式
    with torch.no_grad(): # 禁用梯度计算
        # 使用 tqdm 显示进度
        data_iterator = tqdm(dataloader, desc="Computing Results", leave=False)
        for img, cls, _ in data_iterator:
            clses.append(cls)
            # OrthoHash net returns (logits, codes)
            _, codes = net(img.to(device)) # 只需要 codes (v)
            bs.append(codes.data.cpu()) # 收集连续的 codes
    # 在循环外拼接并计算 sign
    print("Concatenating codes...")
    all_codes_continuous = torch.cat(bs)
    print("Applying sign function...")
    all_codes_binary = all_codes_continuous.sign()
    print("Concatenating labels...")
    all_labels = torch.cat(clses)
    print("Result computation finished.")
    return all_codes_binary, all_labels

# --- 训练和验证函数 (主体结构来自 MyModel_ablation.py) ---
def train_val_orthohash(config, bit):
    # --- 日志和设备设置 ---
    log_dir_suffix = f"{config['dataset']}_{bit}bits"
    log_dir = os.path.join(config["log_dir"], log_dir_suffix)
    writer = SummaryWriter(log_dir=log_dir)
    
    device = config["device"]
    print(f"Using device: {device}")

    # --- 数据加载 (使用你的 sgd_tools.py) ---
    print(f"Loading data using sgd_tools.py for dataset: {config['dataset']}")
    train_loader, test_loader, valid_loader, num_train, num_test, num_valid = get_data(config)
    config["num_train"] = num_train
    config["num_test"] = num_test
    config["num_valid"] = num_valid
    config["batch_num"] = num_train // config["batch_size"] + (1 if num_train % config["batch_size"] != 0 else 0)
    
    print(f"  Training set size: {num_train}")
    print(f"  Test set (Query) size: {num_test}")
    print(f"  Validation set size: {num_valid}")
    print(f"  Database set size (assuming train set is database): {num_train}") # 假设训练集是数据库
    
    # 检查数据格式
    try:
        sample_img, sample_label, _ = next(iter(train_loader))
        is_onehot = sample_label.dim() > 1 and sample_label.size(1) == config['num_classes']
    except Exception as e:
        print(f"  Data loading check failed: {e}")
        return 

    # --- 生成 Codebook ---
    codebook = generate_codebook(config["codebook_method"], config["num_classes"], bit, device=device)

    # --- 模型准备 ---
    original_net_for_backbone = OriginalAlexNet(config["info"], config["hidden_dim"], bit, 1.0) # beta 可能不需要
    backbone_extractor = original_net_for_backbone.feature_layers
    
    net = OrthoHashAlexNetBaseline(backbone_extractor, bit, config["num_classes"], codebook).to(device)

    # --- 优化器准备 ---
    lr_feature = config["optimizer"]["feature_lr"]
    lr_hash = config["optimizer"]["hash_lr"]
    
    parameter_list = [
        {"params": net.get_backbone_params(), "lr": lr_feature},
        {"params": net.get_hash_params(), "lr": lr_hash}
    ]
    
    optimizer = create_optimizer(config, parameter_list)
    print(f"  Optimizer: {config['optimizer_type']}, Nesterov: {config['optimizer']['nesterov']}")
    print(f"  Initial Backbone LR: {optimizer.param_groups[0]['lr']:.2e}, Initial Hash LR: {optimizer.param_groups[1]['lr']:.2e}")

    # --- 损失函数准备 ---
    loss_param = {
        'ce': config['ortho_ce'],
        's': config['ortho_s'],
        'm': config['ortho_m'],
        'm_type': config['ortho_m_type'],
        'device': device 
    }
    criterion = OrthoHashLoss(**loss_param)

    # --- 训练过程变量初始化 ---
    Best_mAP = 0.0
    train_loss_list = []
    time_list = []
    map_list = []
    ap_topK_list = []
    ap_r_list = []
    iter_num = 0
    tst_mAP = 0.0
    tst_AP_topK = 0.0
    tst_AP_r = 0.0
    decay_times = 0 # 初始化学习率衰减次数计数器
    
    # --- 训练循环 ---
    training_start_time = time.time()

    for epoch in range(config["epoch"]):
        print(f"\n--- Epoch {epoch + 1}/{config['epoch']} ---")
        epoch_start_time = time.time()
        net.train()
        train_loss = 0
        
        current_lr = adjust_learning_rate_multistep(optimizer, epoch, config)
        
        # 使用 tqdm 包装 train_loader
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=True)
        for batch_idx, (image, label, ind) in enumerate(batch_iterator):
            image = image.to(device)
            optimizer.zero_grad()
            logits, codes = net(image)
            loss = criterion(logits, codes, label, onehot=is_onehot) 
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_iterator.set_postfix(loss=f"{loss.item():.4f}", refresh=True)
            iter_num += 1
            
        # --- Epoch 结束 ---
        avg_train_loss = train_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)
        epoch_end_time = time.time()
        duration = epoch_end_time - epoch_start_time
        time_list.append(duration)

        if writer:
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('LearningRate/backbone', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('LearningRate/hash', optimizer.param_groups[1]['lr'], epoch)
            writer.add_scalar('EpochTime', duration, epoch)

        # --- 验证部分 ---
        print("--- Validating ---")
        val_binary, val_label = compute_result_orthohash(valid_loader, net, device)
        print(f"  Validation codes computed. Shape: {val_binary.shape}")
        trn_binary, trn_label = compute_result_orthohash(train_loader, net, device)
        print(f"  Database codes computed. Shape: {trn_binary.shape}")
        
        # 计算指标
        if config["dataset"] == "nus-wide":
            mAP = hash_ranking_map_topk(trn_binary.cpu().numpy(), trn_label.cpu().numpy(), 
                                        val_binary.cpu().numpy(), val_label.cpu().numpy(), 
                                        topk=config["topK_mAP"])
        else:
            mAP = hash_ranking_map(trn_binary.cpu().numpy(), trn_label.cpu().numpy(), 
                                   val_binary.cpu().numpy(), val_label.cpu().numpy())
        AP_topK, _ = get_precision_recall_topK(trn_binary.cpu().numpy(), trn_label.cpu().numpy(), 
                                               val_binary.cpu().numpy(), val_label.cpu().numpy(), 
                                               topk=config["topK"])
        AP_r, _ = get_precision_recall_within_hamming_radius(trn_binary.cpu().numpy(), trn_label.cpu().numpy(), 
                                                             val_binary.cpu().numpy(), val_label.cpu().numpy(), 
                                                             r=config["r"])
        print(f"  Validation mAP: {mAP:.4f}, AP@{config['topK']}: {AP_topK:.4f}, AP@R<={config['r']}: {AP_r:.4f}")
        
        if (epoch + 1) % config["log_interval"] == 0:
            if writer:
                writer.add_scalar('mAP/val', mAP, epoch)
                writer.add_scalar('AP_topK/val', AP_topK, epoch)
                writer.add_scalar('AP_r/val', AP_r, epoch)

        # --- 测试与保存最佳模型 ---
        if mAP > Best_mAP:
            Best_mAP = mAP
            print(f"*** New Best Validation mAP: {Best_mAP:.4f} at Epoch {epoch + 1} ***")
            
            print("--- Testing with best model ---")
            tst_binary, tst_label = compute_result_orthohash(test_loader, net, device)
            print(f"  Test codes computed. Shape: {tst_binary.shape}")
            if config["dataset"] == "nus-wide":
                tst_mAP = hash_ranking_map_topk(trn_binary.cpu().numpy(), trn_label.cpu().numpy(), 
                                                tst_binary.cpu().numpy(), tst_label.cpu().numpy(), 
                                                topk=config["topK_mAP"])
            else:
                tst_mAP = hash_ranking_map(trn_binary.cpu().numpy(), trn_label.cpu().numpy(), 
                                           tst_binary.cpu().numpy(), tst_label.cpu().numpy())
            tst_AP_topK, _ = get_precision_recall_topK(trn_binary.cpu().numpy(), trn_label.cpu().numpy(), 
                                                       tst_binary.cpu().numpy(), tst_label.cpu().numpy(), 
                                                       topk=config["topK"])
            tst_AP_r, _ = get_precision_recall_within_hamming_radius(trn_binary.cpu().numpy(), trn_label.cpu().numpy(), 
                                                                     tst_binary.cpu().numpy(), tst_label.cpu().numpy(), 
                                                                     r=config["r"])
            
            # 收集测试结果用于保存
            tst_results = [tst_mAP, tst_AP_topK, tst_AP_r]
            
            if writer:
                writer.add_scalar('mAP/tst_on_best_val', tst_mAP, epoch)
                writer.add_scalar('AP_topK/tst_on_best_val', tst_AP_topK, epoch)
                writer.add_scalar('AP_r/tst_on_best_val', tst_AP_r, epoch)

            # 保存结果文件
            results_path = os.path.join(config["save_path"], f"test_results/{config['dataset']}_{bit}bits")
            os.makedirs(results_path, exist_ok=True)
            results_file = os.path.join(results_path, "test_summary.txt")
            
            if not os.path.exists(results_file):
                with open(results_file, 'w') as f:
                    f.write("Epoch tst_mAP tst_AP_topK tst_AP_r\n")

            with open(results_file, 'a') as f:
                f.write(f"{epoch+1} {tst_mAP:.6f} {tst_AP_topK:.6f} {tst_AP_r:.6f}\n")

            # 保存最佳模型和相关数据
            if epoch >= config["save_epoch_start"]:
                best_model_path = os.path.join(config["save_path"], f'models/{config["dataset"]}_{bit}bits_best_e{epoch+1}_mAP{Best_mAP:.4f}')
                os.makedirs(best_model_path, exist_ok=True)
                torch.save(net.state_dict(), os.path.join(best_model_path, "model.pt"))
                np.save(os.path.join(best_model_path, "val_binary.npy"), val_binary.cpu().numpy())
                np.save(os.path.join(best_model_path, "val_label.npy"), val_label.cpu().numpy())
                np.save(os.path.join(best_model_path, "trn_binary.npy"), trn_binary.cpu().numpy())
                np.save(os.path.join(best_model_path, "trn_label.npy"), trn_label.cpu().numpy())
                np.save(os.path.join(best_model_path, "tst_binary.npy"), tst_binary.cpu().numpy())
                np.save(os.path.join(best_model_path, "tst_label.npy"), tst_label.cpu().numpy())
                np.save(os.path.join(best_model_path, "tst_results.npy"), np.array(tst_results)) 
                print(f"Best model and results saved to: {best_model_path}")
            
        # 记录当前 epoch 的验证指标
        map_list.append(mAP)
        ap_topK_list.append(AP_topK)
        ap_r_list.append(AP_r)

        # 打印当前轮次的最终总结
        print(f"Epoch {epoch + 1}/{config['epoch']} Summary: "
              f"Iter Num: {iter_num}, Decay Times: {decay_times}, "
              f"Bit: {bit}, Dataset: {config['dataset']}, "
              f"Val AP_topK: {AP_topK:.4f}, Val AP_r: {AP_r:.4f}, Val mAP: {mAP:.4f}, "
              f"Best Val mAP: {Best_mAP:.4f}, "
              f"Test mAP (on best): {tst_mAP:.4f}, Test AP_topK (on best): {tst_AP_topK:.4f}, Test AP_r (on best): {tst_AP_r:.4f}, "
              f"Epoch Time: {duration:.2f}s, Train Loss: {avg_train_loss:.4f}")

    # --- 训练结束，保存最终结果 ---
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    print(f"\n--- Training Finished for {bit} bits ---")
    print(f"Total training time: {total_training_time / 3600:.2f} hours")
    print(f"Final Best Validation mAP: {Best_mAP:.6f}")

    final_results_path = os.path.join(config["save_path"], f"final_results/{config['dataset']}_{bit}bits")
    os.makedirs(final_results_path, exist_ok=True)
    
    np.save(os.path.join(final_results_path, "mAP_history.npy"), np.array(map_list))
    np.save(os.path.join(final_results_path, "AP_topK_history.npy"), np.array(ap_topK_list))
    np.save(os.path.join(final_results_path, "AP_r_history.npy"), np.array(ap_r_list))
    np.save(os.path.join(final_results_path, "train_loss_history.npy"), np.array(train_loss_list))
    np.save(os.path.join(final_results_path, "epoch_time_history.npy"), np.array(time_list))
    
    # 保存最终模型权重
    torch.save(net.state_dict(), os.path.join(final_results_path, "model_final.pt"))
    
    # 保存所有配置参数
    config_to_save = config.copy()
    config_to_save['device'] = str(config['device'])
    with open(os.path.join(final_results_path, "config.json"), 'w') as f:
        json.dump(config_to_save, f, indent=4)
        
    print(f"Final results and configuration saved to: {final_results_path}")
    
    if writer:
        writer.close()

# --- 主函数入口 ---
if __name__ == "__main__":
    config = get_orthohash_config()
    
    print("="*30)
    print("       OrthoHash Baseline Runner Configuration       ")
    print("="*30)
    for key, value in config.items():
         if isinstance(value, dict): # 格式化打印字典
             print(f"  {key}:")
             for sub_key, sub_value in value.items():
                 print(f"    {sub_key}: {sub_value}")
         else:
             print(f"  {key}: {value}")
    print("="*30)

    # 对配置中的 bit_list 中每个位数进行训练
    for bit in config["bit_list"]:
        print(f"\n{'='*20} Starting Training for {bit} bits {'='*20}\n")
        train_val_orthohash(config, bit)
        print(f"\n{'='*20} Training for {bit} bits finished {'='*20}\n")

    print("\nAll training finished.")