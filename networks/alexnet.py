

import torch.nn as nn
import torch.nn.init as init
from torchvision import models
from .base_network import BaseHashNetwork

class AlexNet(BaseHashNetwork):
    """
    AlexNet-based hash network
    
    Features:
    - Uses pretrained AlexNet features
    - Adds BatchNorm and ELU activation
    - Two-layer hash head
    """
    
    def _build_feature_layers(self):
        """Build AlexNet feature extraction layers"""
        model_alexnet = models.alexnet(pretrained=self.pretrained)
        
        features = nn.Sequential()
        for i, layer in enumerate(model_alexnet.features):
            if isinstance(layer, nn.Conv2d):
                features.add_module(f"conv_{i}", layer)
                out_channels = layer.out_channels
                features.add_module(f"bn_{i}", nn.BatchNorm2d(out_channels))
                features.add_module(f"elu_{i}", nn.ELU(alpha=1.0, inplace=True))
            elif isinstance(layer, nn.ReLU):
                continue
            else:
                features.add_module(f"layer_{i}", layer)
        
        fc1 = nn.Linear(256 * 6 * 6, 4096)
        fc1_bn = nn.BatchNorm1d(4096)
        fc1.weight = model_alexnet.classifier[1].weight
        fc1.bias = model_alexnet.classifier[1].bias
        
        fc2 = nn.Linear(4096, 4096)
        fc2_bn = nn.BatchNorm1d(4096)
        fc2.weight = model_alexnet.classifier[4].weight
        fc2.bias = model_alexnet.classifier[4].bias
        
        classifier = nn.Sequential(
            fc1, fc1_bn, nn.ELU(alpha=1.0, inplace=True),
            fc2, fc2_bn, nn.ELU(alpha=1.0, inplace=True)
        )
        
        self.feature_layers = nn.Sequential(
            features,
            nn.Flatten(),
            classifier
        )
    
    def _build_hash_layers(self):
        """Build hash layers"""
        fc3 = nn.Linear(4096, self.hidden_dim)
        fc3_bn = nn.BatchNorm1d(self.hidden_dim)
        init.kaiming_normal_(fc3.weight)
        init.constant_(fc3.bias, 0)
        
        fc4 = nn.Linear(self.hidden_dim, self.hash_bits)
        fc4_bn = nn.BatchNorm1d(self.hash_bits)
        init.kaiming_normal_(fc4.weight)
        init.constant_(fc4.bias, 0)
        
        self.hash_layers = nn.Sequential(
            fc3, fc3_bn, nn.ELU(alpha=1.0, inplace=True),
            fc4, fc4_bn
        )