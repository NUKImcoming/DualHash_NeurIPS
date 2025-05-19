import torch.nn as nn
from torchvision import models
import torch.nn.init as init

class AlexNet(nn.Module):
    def __init__(self, info, hidden_dim, hash_bits, beta, pretrained=True):
        super(AlexNet, self).__init__()
        self.info = info
        
        model_alexnet = models.alexnet(pretrained=pretrained)
        self.features = model_alexnet.features
        # for param in self.features.parameters():
        #     param.requires_grad = False
        # print("Freezing feature extraction layers")

        # Process conv layers: pre-trained weights; change loss function; add BN layers: conv-BN-ELU activation-pooling
        # Rebuild features module to insert BN layers and replace activation functions
        self.features = nn.Sequential()
        for i, layer in enumerate(model_alexnet.features):
            if isinstance(layer, nn.Conv2d):
                # Add original conv layer
                self.features.add_module(f"conv_{i}", layer)
                # Get output channels of conv layer, add BN
                out_channels = layer.out_channels
                self.features.add_module(f"bn_{i}", nn.BatchNorm2d(out_channels))
                # Then add ELU activation function
                self.features.add_module(f"elu_{i}", nn.ELU(alpha=1.0, inplace=True))
            elif isinstance(layer, nn.ReLU):
                continue
            else:
                # Keep other layers, such as MaxPool
                self.features.add_module(f"layer_{i}", layer)


        # FC1: BN+pre-trained weights
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc1_bn = nn.BatchNorm1d(4096) 
        self.fc1.weight = model_alexnet.classifier[1].weight
        self.fc1.bias = model_alexnet.classifier[1].bias
        
        #  FC2: BN+pre-trained weights
        self.fc2 = nn.Linear(4096, 4096)
        self.fc2_bn = nn.BatchNorm1d(4096) 
        self.fc2.weight = model_alexnet.classifier[4].weight
        self.fc2.bias = model_alexnet.classifier[4].bias


        # classifier
        self.classifier = nn.Sequential(
            self.fc1,
            self.fc1_bn,
            nn.ELU(alpha=1.0, inplace=True),
            self.fc2,
            self.fc2_bn,
            nn.ELU(alpha=1.0, inplace=True),
        )

        # feature_layers
        self.feature_layers = nn.Sequential(
            self.features,
            nn.Flatten(), 
            self.classifier
        )
        
        # FC3：BN+kaiming initialization
        self.fc3 = nn.Linear(4096, hidden_dim)
        self.fc3_bn = nn.BatchNorm1d(hidden_dim) 
        init.kaiming_normal_(self.fc3.weight)
        init.constant_(self.fc3.bias, 0)
        
        # FC4：BN+kaiming initialization
        self.fc4 = nn.Linear(hidden_dim, hash_bits)
        self.fc4_bn = nn.BatchNorm1d(hash_bits)
        init.kaiming_normal_(self.fc4.weight)
        init.constant_(self.fc4.bias, 0)
        
        # Hash_layer_activation
        self.activation = nn.Tanh()
        self.init_scale = 1.0 
        self.scale = beta
        print("tanh initial beta", beta)

        # hash_layers
        self.hash_layers = nn.Sequential(
            self.fc3,
            self.fc3_bn,
            nn.ELU(alpha=1.0, inplace=True),
            self.fc4,
            self.fc4_bn
        )

    def forward(self, x):
        x = self.feature_layers(x)
        x = self.hash_layers(x)
        x = self.activation(self.scale * x)
        return x