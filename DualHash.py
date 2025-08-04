import numpy as np
import torch
import time
import torch.optim as optim
from networks import create_network  # 使用新的统一网络接口
from data.datasets import get_data  
import os
from Optimizer_B_lambda import *
import lr_schedule
from tqdm import tqdm
import math
from torch.utils.tensorboard import SummaryWriter
from config import * 

torch.multiprocessing.set_sharing_strategy('file_system')

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
    
    print(f"Network: {config['info']}, Dataset: {config['dataset']}, Bits: {bit}")
    
   
    net = create_network(
        network_type=config["net_class"].lower().replace('net', ''),  # 'alexnet' or 'resnet50'
        info=config["info"],
        hidden_dim=config["hidden_dim"],
        hash_bits=bit,
        beta=config["beta"],
        use_tanh=True,
        pretrained=True
    ).to(device)
    
    
    optimizer, lr_scheduler, schedule_param, param_lr = setup_training_components(net, config)
    criterion = DualHashLoss(config)
    
    # 使用新的初始化方法
    B = initialize_B_from_MDSHC_centers(train_loader, bit, device)
    U = torch.zeros(bit, num_train).to(device)
    Z = torch.zeros(bit, num_train).to(device)
    
    Best_mAP = 0.0
    train_losses = []
    training_times = []
    maps = []
    ap_topKs = []
    ap_rs = []                     
    iter_num = 0
    decay_times = 0
    tst_mAP = 0.0
    tst_AP_r = 0.0
    tst_AP_topK = 0.0
    tst_results = []

    for epoch in range(config["epoch"]):
        start = time.time()
        net.train()
        train_loss = 0
        
        for image, label, ind in tqdm(train_loader, leave=False):
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

        if (epoch + 1) % config["log_epoch_interval"] == 0:
            val_binary, val_label = compute_result(valid_loader, net, device=device)
            trn_binary, trn_label = compute_result(train_loader, net, device=device)
            
            # 使用统一的评估函数
            if config["dataset"] == "nus-wide":
                mAP = hash_ranking_map(trn_binary.cpu().numpy(), trn_label.cpu().numpy(), 
                                     val_binary.cpu().numpy(), val_label.cpu().numpy(), 
                                     topk=config["topK_mAP"], dataset_type=config["dataset"])
            else:
                mAP = hash_ranking_map(trn_binary.cpu().numpy(), trn_label.cpu().numpy(), 
                                     val_binary.cpu().numpy(), val_label.cpu().numpy(), 
                                     dataset_type=config["dataset"])
                                     
            AP_topK, _ = get_precision_recall_topK(trn_binary.cpu().numpy(), trn_label.cpu().numpy(), 
                                                 val_binary.cpu().numpy(), val_label.cpu().numpy(), 
                                                 topk=config["topK"])
            AP_r, _ = get_precision_recall_within_hamming_radius(trn_binary.cpu().numpy(), trn_label.cpu().numpy(), 
                                                               val_binary.cpu().numpy(), val_label.cpu().numpy(), 
                                                               r=config["r"])
            
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('mAP/val', mAP, epoch)
            writer.add_scalar('AP_topK/val', AP_topK, epoch)
            writer.add_scalar('AP_r/val', AP_r, epoch)
            
            print(f"Epoch {epoch + 1}/{config['epoch']}: mAP={mAP:.4f}, Best={Best_mAP:.4f}, "
                  f"tst_mAP={tst_mAP:.4f}, loss={train_loss:.4f}, time={duration:.1f}s")

            if mAP > Best_mAP:
                tst_results = []
                Best_mAP = mAP
                tst_binary, tst_label = compute_result(test_loader, net, device=device)
                
                if config["dataset"] == "nus-wide":
                    tst_mAP = hash_ranking_map(trn_binary.cpu().numpy(), trn_label.cpu().numpy(), 
                                             tst_binary.cpu().numpy(), tst_label.cpu().numpy(), 
                                             topk=config["topK_mAP"], dataset_type=config["dataset"])
                else:
                    tst_mAP = hash_ranking_map(trn_binary.cpu().numpy(), trn_label.cpu().numpy(), 
                                             tst_binary.cpu().numpy(), tst_label.cpu().numpy(), 
                                             dataset_type=config["dataset"])
                                             
                tst_results.append(tst_mAP)
                tst_AP_topK, _ = get_precision_recall_topK(trn_binary.cpu().numpy(), trn_label.cpu().numpy(), 
                                                         tst_binary.cpu().numpy(), tst_label.cpu().numpy(), 
                                                         topk=config["topK"])
                tst_results.append(tst_AP_topK)
                tst_AP_r, _ = get_precision_recall_within_hamming_radius(trn_binary.cpu().numpy(), trn_label.cpu().numpy(), 
                                                                       tst_binary.cpu().numpy(), tst_label.cpu().numpy(), 
                                                                       r=config["r"])
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

            maps.append(mAP)
            ap_topKs.append(AP_topK)
            ap_rs.append(AP_r)

    save_path = os.path.join(config["save_path"], f"{config['dataset']}_{bit}bits")
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, "mAP.npy"), maps)
    np.save(os.path.join(save_path, "AP_topK.npy"), ap_topKs)
    np.save(os.path.join(save_path, "AP_r.npy"), ap_rs)
    np.save(os.path.join(save_path, "train_loss.npy"), train_losses)
    np.save(os.path.join(save_path, "time.npy"), training_times)
    
    return tst_mAP, tst_AP_topK, tst_AP_r

def compute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bs.append((net(img.to(device))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar-10', 
                        choices=['cifar-10', 'nus-wide', 'imagenet100'],
                        help='Dataset to use')
    parser.add_argument('--network', type=str, default='alexnet',
                        choices=['alexnet', 'resnet50'],
                        help='Network to use')
    parser.add_argument('--bits', type=int, nargs='+', default=[64],
                        help='Hash bit lengths to test')
    args = parser.parse_args()
    
    # 使用新的配置函数
    config = create_dualhash_config(args.dataset, args.network)
    config["bit_list"] = args.bits
    
    # 设置随机种子
    setup_seed(config["seed"])
    
    print(f"DualHash: {args.network} on {args.dataset}, bits: {args.bits}")

    for bit in config["bit_list"]:
        train_val(config, bit)