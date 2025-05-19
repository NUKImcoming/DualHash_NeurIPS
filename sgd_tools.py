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


class MyCIFAR10(dsets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        target = np.eye(10, dtype=np.int8)[np.array(target)]
        return img, target, index


def cifar_dataset(config):
    batch_size = config["batch_size"]

    train_size = 1000
    test_size = 500
    valid_size = 500 

    transform = transforms.Compose([
        transforms.Resize(config["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    cifar_dataset_root = 'data/cifar-10'

    # Dataset
    train_dataset = MyCIFAR10(root=cifar_dataset_root,
                              train=True,
                              transform=transform,
                              download=True)

    test_dataset = MyCIFAR10(root=cifar_dataset_root,
                             train=False,
                             transform=transform)
    
    valid_dataset = MyCIFAR10(root=cifar_dataset_root,
                              train=True,
                              transform=transform,
                              download=True)
    
    # Concatenate data and labels from both train and test sets
    X = np.concatenate((train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))


    if os.path.exists('data/cifar-10/train_index.npy') and os.path.exists('data/cifar-10/test_index.npy') and os.path.exists('data/cifar-10/valid_index.npy'):
        # Load saved indices
        print("Load saved indices!")
        train_index = np.load('data/cifar-10/train_index.npy')
        test_index = np.load('data/cifar-10/test_index.npy')
        valid_index = np.load('data/cifar-10/valid_index.npy')
    else:
        print("First Load indices!")
        train_index = []
        test_index = []
        valid_index = []

        for label in range(10):
            index = np.where(L == label)[0]

            # Shuffle the indices
            np.random.shuffle(index)

            # Split indices for train, valid, and test sets
            train_index.extend(index[:train_size])
            valid_index.extend(index[train_size:train_size + valid_size])
            test_index.extend(index[train_size + valid_size:train_size + valid_size + test_size])

        # Convert lists to numpy arrays
        train_index = np.array(train_index)
        test_index = np.array(test_index)
        valid_index = np.array(valid_index)

        # Save the indices for future use
        np.save('data/cifar-10/train_index.npy', train_index)
        np.save('data/cifar-10/test_index.npy', test_index)
        np.save('data/cifar-10/valid_index.npy', valid_index)

    # Assign data and targets to corresponding datasets
    train_dataset.data = X[train_index]
    train_dataset.targets = L[train_index]
    test_dataset.data = X[test_index]
    test_dataset.targets = L[test_index]
    valid_dataset.data = X[valid_index]
    valid_dataset.targets = L[valid_index]

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=4)

    return train_loader, test_loader, valid_loader, \
           train_index.shape[0], test_index.shape[0], valid_index.shape[0]


class NusWideDatasetTC21(Dataset):
    """
    Nus-wide dataset, 21 classes.

    Args
        root(str): Path of image files.
        img_txt(str): Path of txt file containing image file name.
        label_txt(str): Path of txt file containing image label.
        transform(callable, optional): Transform images.
        train(bool, optional): Return training dataset.
        num_train(int, optional): Number of training data.
    """
    def __init__(self, root, img_txt, label_txt, transform=None, train=None):
        self.root = root
        self.transform = transform

        img_txt_path = os.path.join(root, img_txt)
        label_txt_path = os.path.join(root, label_txt)

        # Read files
        with open(img_txt_path, 'r') as f:
            self.data = np.array([i.strip() for i in f])
        self.targets = np.loadtxt(label_txt_path, dtype=np.float32)

        # # Sample training dataset
        # if train is True:
        #     perm_index = np.random.permutation(len(self.data))[:num_train]
        #     self.data = self.data[perm_index]
        #     self.targets = self.targets[perm_index]

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

    
    train_dataset = NusWideDatasetTC21(
        root='NUS-WIDE/',
        img_txt='train_img.txt',
        label_txt='train_label_onehot.txt',
        transform=train_transform(),
        train=True
    )
    
    valid_dataset = NusWideDatasetTC21(
    root='NUS-WIDE/',
    img_txt='valid_img.txt',
    label_txt='valid_label_onehot.txt',
    transform=train_transform(),
    train=True
    )
    
    test_dataset = NusWideDatasetTC21(
        root='NUS-WIDE/',
        img_txt='test_img.txt',
        label_txt='test_label_onehot.txt',
        transform=query_transform(),
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        pin_memory=True,
        num_workers=4
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    
    valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=config["batch_size"],
    pin_memory=True,
    num_workers=4
    )
    
    
    # # 初始化计数器
    num_train = len(train_dataset.data)
    num_test = len(test_dataset.data)
    num_valid = len(valid_dataset.data)
    
    
    return train_dataloader, test_dataloader,  valid_dataloader, num_train, num_test, num_valid


def get_data(config):
    
    if config["dataset"] == "cifar-10":
        return cifar_dataset(config)
    elif config["dataset"] == "nus-wide":
        return nus_wide_dataset(config)


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

# mAP, topk=-1
def hash_ranking_map(retrieval_codes, retrieval_labels, query_codes, query_labels):
    num_query = query_codes.shape[0]
    mAP = 0.0

    # 计算地面真实标签矩阵
    ground_truth = (np.dot(query_labels, retrieval_labels.T) > 0).astype(np.float32)
    # 计算汉明距离
    hamming_dist = CalcHammingDist(query_codes, retrieval_codes)
    # 对汉明距离进行排序，获得索引
    sorted_indices = np.argsort(hamming_dist, axis=1)
    count_valid_query = 0

    for i in range(num_query):
        gnd = ground_truth[i]
        relevant_num = np.sum(gnd).astype(int)
        if relevant_num == 0:
            continue
        gnd = gnd[sorted_indices[i]]
        pos_score = np.linspace(1, relevant_num, relevant_num)
        relevant_indices = np.nonzero(gnd)[0].astype(np.float32) + 1
        mAP += np.mean(pos_score / relevant_indices)
        count_valid_query += 1

    if count_valid_query != 0:
        mAP /= count_valid_query
    else:
        print(f"查询集的有效检索数为{count_valid_query}，请检查模型或数据集")

    return mAP


def hash_ranking_map_topk(retrieval_codes, retrieval_labels, query_codes, query_labels, topk=5000):
    num_query = query_labels.shape[0]
    num_gallery = retrieval_labels.shape[0]
    topkmap = 0

    for iter in range(num_query):
        gnd = (np.dot(query_labels[iter, :], retrieval_labels.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(query_codes[iter, :], retrieval_codes)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_

    topkmap = topkmap / num_query

    return topkmap



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