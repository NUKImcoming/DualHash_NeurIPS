import torch.nn as nn
import torch.nn.init as init
from torchvision import models
from .base_network import BaseNetwork

class ResNet50(BaseNetwork):
    """
    ResNet50-based hash network (simplified version)
    
    Features:
    - Uses pretrained ResNet50 features
    - Two-layer hash head (2048->hidden_dim->hash_bits)
    """
    
    def _build_feature_layers(self):
        """Build ResNet50 feature extraction layers"""
        model_resnet50 = models.resnet50(pretrained=self.pretrained)
        
        features = nn.Sequential(*list(model_resnet50.children())[:-2])
        
        adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.feature_layers = nn.Sequential(
            features,
            adaptive_pool,
            nn.Flatten()
        )
    
    def _build_hash_layers(self):
        """Build hash layers"""
        resnet_feature_dim = 2048
        
        fc1 = nn.Linear(resnet_feature_dim, self.hidden_dim)
        fc1_bn = nn.BatchNorm1d(self.hidden_dim)
        init.kaiming_normal_(fc1.weight)
        init.constant_(fc1.bias, 0)
        
        fc2 = nn.Linear(self.hidden_dim, self.hash_bits)
        fc2_bn = nn.BatchNorm1d(self.hash_bits)
        init.kaiming_normal_(fc2.weight)
        init.constant_(fc2.bias, 0)
        
        self.hash_layers = nn.Sequential(
            fc1, fc1_bn, nn.ELU(alpha=1.0, inplace=True),
            fc2, fc2_bn
        )

class ResNet50_v2(BaseNetwork):
    """
    Enhanced ResNet50 version - more similar to AlexNet structure
    
    Features:
    - Contains more hidden layers to match AlexNet complexity
    - Three-layer hash head (2048->4096->hidden_dim->hash_bits)
    """
    
    def _build_feature_layers(self):
        """Build ResNet50 feature extraction layers"""
        model_resnet50 = models.resnet50(pretrained=self.pretrained)
        
        features = nn.Sequential(*list(model_resnet50.children())[:-2])
        
        adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.feature_layers = nn.Sequential(
            features,
            adaptive_pool,
            nn.Flatten()
        )
    
    def _build_hash_layers(self):
        """Build multi-layer hash layers"""
        resnet_feature_dim = 2048
        
        fc1 = nn.Linear(resnet_feature_dim, 4096)
        fc1_bn = nn.BatchNorm1d(4096)
        init.kaiming_normal_(fc1.weight)
        init.constant_(fc1.bias, 0)
        
        fc2 = nn.Linear(4096, self.hidden_dim)
        fc2_bn = nn.BatchNorm1d(self.hidden_dim)
        init.kaiming_normal_(fc2.weight)
        init.constant_(fc2.bias, 0)
        
        fc3 = nn.Linear(self.hidden_dim, self.hash_bits)
        fc3_bn = nn.BatchNorm1d(self.hash_bits)
        init.kaiming_normal_(fc3.weight)
        init.constant_(fc3.bias, 0)
        
        self.hash_layers = nn.Sequential(
            fc1, fc1_bn, nn.ELU(alpha=1.0, inplace=True),
            fc2, fc2_bn, nn.ELU(alpha=1.0, inplace=True),
            fc3, fc3_bn
        )