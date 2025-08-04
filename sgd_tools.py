import numpy as np
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as dsets
import os
import json
from transform import train_transform, query_transform
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset



def compute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bs.append((net(img.to(device))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

# # mAP, topk=-1 For cifar10
# def hash_ranking_map(retrieval_codes, retrieval_labels, query_codes, query_labels):
#     num_query = query_codes.shape[0]
#     mAP = 0.0

#     # 计算地面真实标签矩阵
#     ground_truth = (np.dot(query_labels, retrieval_labels.T) > 0).astype(np.float32)
#     # 计算汉明距离
#     hamming_dist = CalcHammingDist(query_codes, retrieval_codes)
#     # 对汉明距离进行排序，获得索引
#     sorted_indices = np.argsort(hamming_dist, axis=1)
#     count_valid_query = 0

#     for i in range(num_query):
#         gnd = ground_truth[i]
#         relevant_num = np.sum(gnd).astype(int)
#         if relevant_num == 0:
#             continue
#         gnd = gnd[sorted_indices[i]]
#         pos_score = np.linspace(1, relevant_num, relevant_num)
#         relevant_indices = np.nonzero(gnd)[0].astype(np.float32) + 1
#         mAP += np.mean(pos_score / relevant_indices)
#         count_valid_query += 1

#     if count_valid_query != 0:
#         mAP /= count_valid_query
#     else:
#         print(f"查询集的有效检索数为{count_valid_query}，请检查模型或数据集")

#     return mAP

# # for nus-wide
# def hash_ranking_map_topk(retrieval_codes, retrieval_labels, query_codes, query_labels, topk=5000):
#     num_query = query_labels.shape[0]
#     num_gallery = retrieval_labels.shape[0]
#     topkmap = 0

#     for iter in range(num_query):
#         gnd = (np.dot(query_labels[iter, :], retrieval_labels.transpose()) > 0).astype(np.float32)
#         hamm = CalcHammingDist(query_codes[iter, :], retrieval_codes)
#         ind = np.argsort(hamm)
#         gnd = gnd[ind]

#         tgnd = gnd[0:topk]
#         tsum = np.sum(tgnd).astype(int)
#         if tsum == 0:
#             continue
#         count = np.linspace(1, tsum, tsum)

#         tindex = np.asarray(np.where(tgnd == 1)) + 1.0
#         topkmap_ = np.mean(count / (tindex))
#         topkmap = topkmap + topkmap_

#     topkmap = topkmap / num_query

#     return topkmap


def hash_ranking_map(retrieval_codes, retrieval_labels, query_codes, query_labels, topk=-1, dataset_type="cifar-10"):

    num_query = query_codes.shape[0]
    num_gallery = retrieval_codes.shape[0]
    
    # 考虑数据集特点决定是否使用topk
    if dataset_type.lower() == "nus-wide":
        # NUS-WIDE使用topk，默认5000
        if topk <= 0:
            topk = 5000
    else:
        # CIFAR-10默认使用全部结果
        if topk <= 0:
            topk = num_gallery
    
    # 限制topk不超过检索集大小
    topk = min(topk, num_gallery)
    
    # 计算地面真实标签矩阵
    ground_truth = (np.dot(query_labels, retrieval_labels.T) > 0).astype(np.float32)
    
    # 预计算全部汉明距离并排序 (CIFAR-10风格，更高效)
    hamming_dist = CalcHammingDist(query_codes, retrieval_codes)
    sorted_indices = np.argsort(hamming_dist, axis=1)
    
    mAP = 0.0
    count_valid_query = 0
    
    for i in range(num_query):
        # 获取排序后的相关性
        gnd = ground_truth[i][sorted_indices[i]]
        
        # 只考虑前topk个结果
        tgnd = gnd[:topk]
        tsum = np.sum(tgnd).astype(int)
        
        # 如果没有相关项，跳过
        if tsum == 0:
            continue
            
        # 为相关项分配分数
        count = np.linspace(1, tsum, tsum)
        
        # 找出相关项位置
        relevant_indices = np.nonzero(tgnd)[0].astype(np.float32) + 1
        
        # 计算当前查询的AP
        query_map = np.mean(count / relevant_indices)
        mAP += query_map
        count_valid_query += 1
    
    # 计算平均值
    if count_valid_query != 0:
        mAP /= count_valid_query
    else:
        print(f"查询集的有效检索数为{count_valid_query}，请检查模型或数据集")
    
    return mAP

# faster but more memory
def CalcTopMapWithPR(qB, queryL, rB, retrievalL, topk):
    num_query = queryL.shape[0]
    num_gallery = retrievalL.shape[0]
    topkmap = 0
    prec = np.zeros((num_query, num_gallery))
    recall = np.zeros((num_query, num_gallery))
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        all_sim_num = np.sum(gnd)

        prec_sum = np.cumsum(gnd)
        return_images = np.arange(1, num_gallery + 1)

        prec[iter, :] = prec_sum / return_images
        recall[iter, :] = prec_sum / all_sim_num

        assert recall[iter, -1] == 1.0
        assert all_sim_num == prec_sum[-1]

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    index = np.argwhere(recall[:, -1] == 1.0)
    index = index.squeeze()
    prec = prec[index]
    recall = recall[index]
    cum_prec = np.mean(prec, 0)
    cum_recall = np.mean(recall, 0)

    return topkmap

# topK-precision-recall
def get_precision_recall_topK(retrieval_codes, retrieval_labels, query_codes, query_labels, topk=1000):
    num_query, _  = query_codes.shape
    num_retrieval, _ = retrieval_codes.shape
    precision_topk = 0.0
    recall_topk = 0.0
    
    P = []
    R = []

     # 计算地面真实标签矩阵
    ground_truth = (np.dot(query_labels, retrieval_labels.T) > 0).astype(np.float32)
    # 计算汉明距离
    hamming_dist = CalcHammingDist(query_codes, retrieval_codes)
    # 对汉明距离进行排序，获得索引
    sorted_indices = np.argsort(hamming_dist, axis=1)
    count_valid_query = 0

    for i in tqdm(range(num_query)):
        # 每个query的真实相似度检索：(num_retr, )
        gnd = ground_truth[i]
        # 真实相关总数
        gnd_relevant_num = np.sum(gnd).astype(int)
        if gnd_relevant_num == 0:
            continue
        count_valid_query += 1
        sorted_indices_topk = sorted_indices[i, :topk]
        gnd_topK = gnd[sorted_indices_topk]
        gnd_relevant_num_topk = np.sum(gnd_topK).astype(int)
        if gnd_relevant_num_topk == 0:
            continue
        
        if topk == -1:
            P.append(gnd_relevant_num_topk / num_retrieval)
        else:
            P.append(gnd_relevant_num_topk / topk)
        R.append(gnd_relevant_num_topk / gnd_relevant_num)

    precision_topk = np.sum(P) / count_valid_query
    recall_topk = np.sum(R) / count_valid_query

    return precision_topk.item(), recall_topk.item()

# precision-recall@r
def get_precision_recall_within_hamming_radius(retr_codes, retr_labels, qury_codes, qury_labels, r):
    num_qury, bits = qury_codes.shape

    # 初始化精确度和召回率数组
    P = np.zeros(num_qury)
    R = np.zeros(num_qury)
    
    # 计算汉明距离
    hamming_dist = CalcHammingDist(qury_codes, retr_codes) 

    # 真实的相似度：利用标签计算，并且将bool变量转换成float
    ground_truth = (np.dot(qury_labels, retr_labels.T) > 0).astype(float)

    # 有效查询
    count_valid_query = 0

    for i in range(num_qury):
        gnd = ground_truth[i]
        gnd_relevant_num = np.sum(gnd)

        if gnd_relevant_num == 0:
            continue

        retr_dist_i = hamming_dist[i]
        count_retr_radius_i = np.sum(retr_dist_i <= r)
        count_valid_query += 1

        if count_retr_radius_i == 0:
            continue
        

        tmq = gnd * (retr_dist_i <= r)
        count_pos_radius_i = np.sum(tmq)
        
        P[i] = count_pos_radius_i / count_retr_radius_i
        R[i] = count_pos_radius_i / gnd_relevant_num
    
    if count_valid_query > 0:
        Precision_radius = np.sum(P) / count_valid_query
        Recall_radius = np.sum(R) / count_valid_query
    else:
        Precision_radius = 0.0
        Recall_radius = 0.0

    return Precision_radius, Recall_radius


def initialize_B_with_PCA(train_loader, net, bit, device):
    """Initialize binary codes using PCA"""
    print("Initializing binary codes with PCA...")
    
    features = []
    all_indices = []
    
    net.eval()
    with torch.no_grad():
        for image, _, ind in tqdm(train_loader):
            image = image.to(device)
            x = net.feature_layers(image)
            features.append(x.cpu())
            all_indices.append(ind)
    
    features = torch.cat(features, dim=0)
    all_indices = torch.cat(all_indices)
    
    features = features - features.mean(dim=0, keepdim=True)
    
    try:
        if features.shape[0] > 5000:
            idx = torch.randperm(features.shape[0])[:5000]
            sample_features = features[idx]
        else:
            sample_features = features
            
        U, S, V = torch.svd(sample_features.t())
        projection = U[:, :bit]
        
        projected = torch.mm(features, projection)
        
        B = torch.zeros(bit, len(train_loader.dataset)).to(device)
        for i, idx in enumerate(all_indices):
            B[:, idx] = projected[i].t()
        
        return torch.sign(B)
        
    except:
        return torch.randn(bit, len(train_loader.dataset)).sign().to(device)


def initialize_B_with_ITQ(train_loader, net, bit, device, n_iter=50):
    """Initialize binary codes using Iterative Quantization"""
    print("Initializing binary codes with ITQ...")
    
    features = []
    all_indices = []
    
    net.eval()
    with torch.no_grad():
        for image, _, ind in tqdm(train_loader):
            image = image.to(device)
            x = net.feature_layers(image)
            features.append(x.cpu())
            all_indices.append(ind)
    
    features = torch.cat(features, dim=0)
    all_indices = torch.cat(all_indices)
    
    try:
        features = features - features.mean(dim=0, keepdim=True)
        
        if features.shape[0] > 10000:
            sample_size = 10000
            indices = torch.randperm(features.shape[0])[:sample_size]
            sample_features = features[indices]
            cov = torch.mm(sample_features.t(), sample_features) / sample_size
        else:
            cov = torch.mm(features.t(), features) / features.shape[0]
        
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        sorted_indices = torch.argsort(eigenvalues, descending=True)[:bit]
        projection = eigenvectors[:, sorted_indices]
        
        V = torch.mm(features, projection)
        
        R = torch.randn(bit, bit)
        U, _, VT = torch.linalg.svd(R)
        R = torch.mm(U, VT)
        
        for i in range(n_iter):
            B = torch.sign(torch.mm(V, R))
            C = torch.mm(V.t(), B)
            UB, _, VB = torch.linalg.svd(C)
            R = torch.mm(UB, VB)
        
        final_B = torch.sign(torch.mm(V, R))
        
        B = torch.zeros(bit, len(train_loader.dataset))
        for i, idx in enumerate(all_indices):
            B[:, idx] = final_B[i].t()
        
        return B.to(device)
        
    except:
        B = torch.zeros(bit, len(train_loader.dataset)).to(device)
        
        for i in range(bit):
            rand_perm = torch.randperm(B.shape[1])
            half_point = B.shape[1] // 2
            B[i, rand_perm[:half_point]] = 1
            B[i, rand_perm[half_point:]] = -1
            
        return B
    
def initialize_B_from_MDSHC_centers(train_loader, bit, device, centers_path="./data/centers_100_32.npy"):
 
    mdshc_centers = np.load(centers_path)
    mdshc_centers = torch.from_numpy(mdshc_centers).float().to(device)
    
    num_samples = len(train_loader.dataset)
    B = torch.zeros(bit, num_samples).to(device)
    
    for _, label, ind in tqdm(train_loader, desc="B矩阵初始化"):
        label = label.to(device)
        for i, idx in enumerate(ind):
            # one-hot转类别ID
            class_id = torch.argmax(label[i]).item()
            B[:, idx] = mdshc_centers[class_id, :]
    
    print(f"B矩阵初始化完成，使用MDSHC中心")
    return B

def detect_plateau_and_adjust_lr(optimizer, train_losses, epoch, patience=5, factor=10):
    if len(train_losses) < patience:
        return False
    
    # 看最近5轮loss是否几乎不变
    recent_losses = train_losses[-patience:]
    improvement = recent_losses[0] - recent_losses[-1]  # 首尾差值
    
    if improvement < 1e-4:  # 改善太小
        # 提高学习率跳出停滞
        for param_group in optimizer.param_groups:
            param_group['lr'] *= factor
        return True