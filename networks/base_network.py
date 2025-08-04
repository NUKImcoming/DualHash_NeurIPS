
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseNetwork(nn.Module, ABC):

    
    def __init__(self, info="", hidden_dim=1024, hash_bits=64, beta=1.0, 
                 use_tanh=True, pretrained=True):
        """
        Initialize base network
        
        Args:
            info (str): Network information identifier
            hidden_dim (int): Hidden layer dimension
            hash_bits (int): Number of hash bits
            beta (float): Scaling factor for Tanh activation
            use_tanh (bool): Whether to use Tanh activation
            pretrained (bool): Whether to use pretrained weights
        """
        super(BaseNetwork, self).__init__()
        self.info = info
        self.hidden_dim = hidden_dim
        self.hash_bits = hash_bits
        self.beta = beta
        self.use_tanh = use_tanh
        self.pretrained = pretrained
        
        self.feature_layers = None
        self.hash_layers = None
        self.activation = None
        
        self._build_network()
    
    @abstractmethod
    def _build_feature_layers(self):
        """Build feature extraction layers - subclasses must implement"""
        pass
    
    @abstractmethod
    def _build_hash_layers(self):
        """Build hash layers - subclasses must implement"""
        pass
    
    def _build_activation(self):
        """Build activation function"""
        if self.use_tanh:
            self.activation = nn.Tanh()
            self.scale = self.beta
            print(f"{self.__class__.__name__} using Tanh activation, beta={self.beta}")
        else:
            self.activation = nn.Identity()
            self.scale = 1.0
            print(f"{self.__class__.__name__} not using activation function")
    
    def _build_network(self):
        """Build complete network"""
        self._build_feature_layers()
        self._build_hash_layers()
        self._build_activation()
    
    def forward(self, x):
        """Forward pass"""
        x = self.feature_layers(x)
        x = self.hash_layers(x)
        
        if self.use_tanh:
            x = self.activation(self.scale * x)
        else:
            x = self.activation(x)
        
        return x
    
    def get_features(self, x):
        """Get feature representation (without hash layers)"""
        return self.feature_layers(x)
    
    def get_hash_codes(self, x):
        """Get hash codes (complete forward pass)"""
        return self.forward(x)
    
    def freeze_feature_layers(self):
        """Freeze feature extraction layer parameters"""
        for param in self.feature_layers.parameters():
            param.requires_grad = False
        print("Feature extraction layer parameters frozen")
    
    def unfreeze_feature_layers(self):
        """Unfreeze feature extraction layer parameters"""
        for param in self.feature_layers.parameters():
            param.requires_grad = True
        print("Feature extraction layer parameters unfrozen")