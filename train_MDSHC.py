import torch
import torch.optim as optim
import os
import time
import numpy as np
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
from MDSHC_network import MoCoAlexNet
from MDSHC_loss import MDSHCLossAlexNet
from sgd_tools import *

def compute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        # 只保留第一个返回值（哈希码）
        bs.append(net(img.to(device))[0].data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train_val_mdshc(config, bit, l):
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
    
    # 设置哈希中心路径
    # config["center_path"] = f"./centerswithoutVar/centers_{config['n_class']}_{bit}.npy"
    center_path = f"./centerswithoutVar/centers_{config['n_class']}_{bit}.npy"
    config["center_path"] = center_path
    
    
    # 模型实例化
    print("初始化模型...")
    net = MDSHCAlexNet(
        hidden_dim=config["hidden_dim"], 
        hash_bits=bit,
        pretrained=True
    ).to(device)
    
    # 损失函数实例化
    print("设置损失函数...")
    criterion = MDSHCLossAlexNet(config, bit, l)
    
    # 优化器设置
    print("设置优化器...")
    # 分离参数组 - 特征提取部分和哈希部分
    feature_lr = config["optimizer"]["feature_lr"]
    hash_lr = config["optimizer"]["hash_lr"]
    
    # 分离参数组
    feature_params = net.feature_extractor.parameters()
    hash_params = net.hash_layer.parameters()
    
    parameter_list = [
        {"params": feature_params, "lr": feature_lr},
        {"params": hash_params, "lr": hash_lr}
    ]
    
    # 创建优化器
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
    ar_topK_list = []
    ap_r_list = []
    ar_r_list = []
    
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
        
        # 调整学习率
        milestones = [int(m.strip()) for m in config["lr_scheduler_params"]["milestones"].split(',')]
        gamma = config["lr_scheduler_params"]["gamma"]
        
        # 计算衰减因子
        decay_factor = gamma ** sum(1 for m in milestones if epoch >= m)
        
        # 应用学习率调整
        optimizer.param_groups[0]['lr'] = feature_lr * decay_factor
        optimizer.param_groups[1]['lr'] = hash_lr * decay_factor
        
        # 训练一个epoch
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1} 训练", leave=True)
        for batch_idx, (image, label, ind) in enumerate(batch_iterator):
            image = image.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播 - 注意这里只使用MDSHCAlexNet的第一个输出
            hash_code, _ = net(image)
            
            # 计算损失
            loss, center_loss, pair_loss = criterion(hash_code, label, ind, epoch)
            
            # 反向传播和优化
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)  # 添加合理的梯度裁剪值
            optimizer.step()
            
            # 累加损失
            train_loss += loss.item()
            train_center_loss += center_loss.item()
            if epoch >= config['ggss_params']['epoch_change']:
                train_pair_loss += pair_loss.item()
            
            # 更新进度条
            batch_iterator.set_postfix({
                'loss': f"{loss.item():.4f}",
                'center': f"{center_loss.item():.4f}",
                'pair': f"{pair_loss.item():.4f}" if epoch >= config['ggss_params']['epoch_change'] else "0.0000"
            })
        
        # 计算平均损失
        avg_train_loss = train_loss / len(train_loader)
        avg_center_loss = train_center_loss / len(train_loader)
        avg_pair_loss = train_pair_loss / len(train_loader)
        
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
            mAP = hash_ranking_map(trn_binary.cpu().numpy(), trn_label.cpu().numpy(), 
                                  val_binary.cpu().numpy(), val_label.cpu().numpy())
            
            AP_topK, AR_topK = get_precision_recall_topK(
                trn_binary.cpu().numpy(), trn_label.cpu().numpy(),
                val_binary.cpu().numpy(), val_label.cpu().numpy(),
                topk=config["topK"]
            )
            
            AP_r, AR_r = get_precision_recall_within_hamming_radius(
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
                
                tst_mAP = hash_ranking_map(
                    trn_binary.cpu().numpy(), trn_label.cpu().numpy(),
                    tst_binary.cpu().numpy(), tst_label.cpu().numpy()
                )
                
                tst_AP_topK, tst_AR_topK = get_precision_recall_topK(
                    trn_binary.cpu().numpy(), trn_label.cpu().numpy(),
                    tst_binary.cpu().numpy(), tst_label.cpu().numpy(),
                    topk=config["topK"]
                )
                
                tst_AP_r, tst_AR_r = get_precision_recall_within_hamming_radius(
                    trn_binary.cpu().numpy(), trn_label.cpu().numpy(),
                    tst_binary.cpu().numpy(), tst_label.cpu().numpy(),
                    r=config["r"]
                )
                
                # 收集测试结果
                tst_results = [tst_mAP, tst_AP_topK, tst_AP_r, tst_AR_topK, tst_AR_r]
                
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
                        f.write("Epoch tst_mAP tst_AP_topK tst_AP_r tst_AR_topK tst_AR_r\n")
                
                with open(results_file, 'a') as f:
                    f.write(f"{epoch+1} {tst_mAP:.6f} {tst_AP_topK:.6f} {tst_AP_r:.6f} {tst_AR_topK:.6f} {tst_AR_r:.6f}\n")
                
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
            ar_topK_list.append(AR_topK)
            ap_r_list.append(AP_r)
            ar_r_list.append(AR_r)
            
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
    np.save(os.path.join(final_results_path, "AR_topK_history.npy"), np.array(ar_topK_list))
    np.save(os.path.join(final_results_path, "AP_r_history.npy"), np.array(ap_r_list))
    np.save(os.path.join(final_results_path, "AR_r_history.npy"), np.array(ar_r_list))
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