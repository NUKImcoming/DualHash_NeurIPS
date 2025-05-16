import torch.nn as nn
from torchvision import models
import torch.nn.init as init
import torch.nn.functional as F
import torch



class AlexNet(nn.Module):
    def __init__(self, info, hidden_dim, hash_bits, beta, pretrained=True):
        super(AlexNet, self).__init__()
        self.info = info
        
        model_alexnet = models.alexnet(pretrained=pretrained)
        self.features = model_alexnet.features
        # for param in self.features.parameters():
        #     param.requires_grad = False
        # print("冻结特征提取层")

        # 处理卷积层：pre-trained weights；更换损失函数；添加BN层：卷积-BN-ELU激活-池化
        # 重新构建 features 模块，以插入 BN 层和替换激活函数
        self.features = nn.Sequential()
        for i, layer in enumerate(model_alexnet.features):
            if isinstance(layer, nn.Conv2d):
                # 添加原始卷积层
                self.features.add_module(f"conv_{i}", layer)
                # 获取卷积层的输出通道数，添加 BN
                out_channels = layer.out_channels
                self.features.add_module(f"bn_{i}", nn.BatchNorm2d(out_channels))
                # 接着添加 ELU 激活函数
                self.features.add_module(f"elu_{i}", nn.ELU(alpha=1.0, inplace=True))
            elif isinstance(layer, nn.ReLU):
                # 忽略原始的 ReLU，因为我们已添加 ELU
                continue
            else:
                # 保留其它层，如 MaxPool
                self.features.add_module(f"layer_{i}", layer)


        # 全连接层1：BN+pre-trained weights
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc1_bn = nn.BatchNorm1d(4096)  # 添加批量归一化层
        self.fc1.weight = model_alexnet.classifier[1].weight
        self.fc1.bias = model_alexnet.classifier[1].bias
        
        # 全连接层2: BN+pre-trained weights
        self.fc2 = nn.Linear(4096, 4096)
        self.fc2_bn = nn.BatchNorm1d(4096)  # 添加批量归一化层
        self.fc2.weight = model_alexnet.classifier[4].weight
        self.fc2.bias = model_alexnet.classifier[4].bias


        # 构建分类器层
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            self.fc1,
            self.fc1_bn,
            nn.ELU(alpha=1.0, inplace=True),
            # nn.Dropout(),
            self.fc2,
            self.fc2_bn,
            nn.ELU(alpha=1.0, inplace=True),
        )

        # 汇总特征提取层：
        self.feature_layers = nn.Sequential(
            self.features,
            nn.Flatten(),  # 确保features到classifier的正确过渡
            self.classifier
        )
        
        # 全连接层3：BN+kaiming初始化
        self.fc3 = nn.Linear(4096, hidden_dim)
        self.fc3_bn = nn.BatchNorm1d(hidden_dim) 
        init.kaiming_normal_(self.fc3.weight)
        init.constant_(self.fc3.bias, 0)
        
        # 全连接层4：BN+kaiming初始化
        self.fc4 = nn.Linear(hidden_dim, hash_bits)
        self.fc4_bn = nn.BatchNorm1d(hash_bits)
        init.kaiming_normal_(self.fc4.weight)
        init.constant_(self.fc4.bias, 0)
        
        # 哈希激活
        self.activation = nn.Tanh()
        self.init_scale = 1.0 
        if "DDSH" in self.info :
            self.scale = beta
            print("tanh 的 初始系数是：", beta)
        else:
            self.scale = self.init_scale

        # 哈希层
        self.hash_layers = nn.Sequential(
            # nn.Dropout(),
            self.fc3,
            self.fc3_bn,
            nn.ELU(alpha=1.0, inplace=True),
            self.fc4,
            self.fc4_bn
        )

    def forward(self, x):
        x = self.feature_layers(x)
        x = self.hash_layers(x)
        if "HashNet" in self.info or "DHN" in self.info or "MLP" in self.info or "DDSH" in self.info:
            x = self.activation(self.scale * x)
        return x
