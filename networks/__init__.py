
from .base_network import BaseHashNetwork
from .alexnet import AlexNet
from .resnet import ResNet50,  ResNet50V2

def create_network(network_type, info="", hidden_dim=1024, hash_bits=64, 
                  beta=1.0, use_tanh=True, pretrained=True, version="v1"):
    """

    
    Args:
        network_type (str): Network type ['alexnet', 'resnet50']
        info (str): Network information identifier
        hidden_dim (int): Hidden layer dimension
        hash_bits (int): Number of hash bits
        beta (float): Scaling factor for Tanh activation
        use_tanh (bool): Whether to use Tanh activation
        pretrained (bool): Whether to use pretrained weights
        version (str): Network version ['v1', 'v2'] (only applies to resnet50)
        
    Returns:
        nn.Module: Corresponding network instance
    """
    networks = {
        'alexnet': AlexNet,
        'resnet50': {
            'v1': ResNet50,
            'v2': ResNet50V2
        }
    }
    
    if network_type not in networks:
        raise ValueError(f"Unsupported network type: {network_type}")
    
    # Handle alexnet (single version)
    if network_type == 'alexnet':
        network_class = networks[network_type]
    else:
        # Handle resnet50 with versions
        if version not in networks[network_type]:
            raise ValueError(f"Network {network_type} does not support version: {version}")
        network_class = networks[network_type][version]
    
    return network_class(
        info=info,
        hidden_dim=hidden_dim,
        hash_bits=hash_bits,
        beta=beta,
        use_tanh=use_tanh,
        pretrained=pretrained
    )

__all__ = [
    'BaseHashNetwork',
    'AlexNet',
    'ResNet50', 'ResNet50V2',
    'create_network'
]
