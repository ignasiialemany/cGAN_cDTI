import torch 
import torch.nn as nn
import torch.optim as optim
import yaml
import numpy as np
from utils import load_config
import torch.nn.functional as F

def init_weights(m, mean=0.0, std=0.02):
    """Initialize network weights with Gaussian distribution."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        nn.init.normal_(m.weight.data, mean=mean, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, mean=1.0, std=std)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, mean=mean, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config        
        self.encoder = Encoder(config["generator"]["in_channels"], config["generator"]["features"], 
                               config["generator"]["kernel_size"], config["generator"]["stride"], 
                               config["generator"]["noise_dim"], config["generator"]["padding"])
        self.decoder = Decoder(config["generator"]["in_channels"],config["generator"]["features"], 
                               config["generator"]["kernel_size"], config["generator"]["stride"],
                               config["generator"]["noise_dim"], config["generator"]["padding"])
        
        # Initialize weights with Gaussian distribution
        self.apply(lambda m: init_weights(m, std=0.02))
        
    def forward(self, x):
        # Store the mask of white regions for attention
        x, skip_connections = self.encoder(x)
        x = self.decoder(x, skip_connections)
        return x
    
class Encoder(nn.Module):
    def __init__(self, in_channel, features, kernel_size, stride, noise_dim, padding):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        self.features = features
        
        # 1-64 block, 64-128 block, 128-256 block, 256-512 block
        for i in range(len(features)):
            layers = []
            # First layer should handle the input channels correctly
            if i == 0:
                # This is the first layer - use in_channel from config
                layers.append(nn.Conv2d(in_channel, features[i], kernel_size, stride, padding))
            else:
                # Subsequent layers use the output of previous layer as input
                layers.append(nn.Conv2d(features[i-1], features[i], kernel_size, stride, padding))
            
            # Only to first block
            if i > 0:
                layers.append(nn.BatchNorm2d(features[i]))
            layers.append(nn.LeakyReLU(0.2))
            self.encoder_blocks.append(nn.Sequential(*layers))
        
        
    def forward(self, x):
        skip_connections = []
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            skip_connections.append(x)
        return x, skip_connections


class Decoder(nn.Module):
    def __init__(self, input_features, features, kernel_size, stride, noise_dim, padding):
        super().__init__()
        self.decoder_blocks = nn.ModuleList()
        
        # 512+noise_dim - 256 block, 256-128 block, 128-64 block, 64-1 block
        for i in range(len(features)-1, 0, -1):
            layers = []
            if i == len(features)-1:
                layers.append(nn.ConvTranspose2d(features[i]+noise_dim, features[i-1], kernel_size, stride, padding))
            else:
                layers.append(nn.ConvTranspose2d(features[i]*2, features[i-1], kernel_size, stride, padding))                
            layers.append(nn.BatchNorm2d(features[i-1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.6))
            #layers.append(nn.Dropout(0.05))
            self.decoder_blocks.append(nn.Sequential(*layers))
            
            # Add attention blocks
            #if i < len(features)-1:
                #self.attention_blocks.append(SpatialAttention(features[i-1]))
        
        # Final layer
        self.decoder_blocks.append(nn.Sequential(
            nn.ConvTranspose2d(features[0]*2, input_features, kernel_size, stride, padding),
            nn.Sigmoid()
        ))
        
    def forward(self, x, skip_connections):
        skip_connections = skip_connections[::-1]
        x = self.decoder_blocks[0](x)
        
        for i in range(1, len(self.decoder_blocks)):
            # Concat with skip connection
            x = torch.cat([x, skip_connections[i]], dim=1)
            # Apply decoder block
            x = self.decoder_blocks[i](x)  
            # Normalize to [0,1]
            # x = (x+1)/2      
        return x
        
if __name__ == "__main__":
    
    config = load_config("config.yaml")
    print(config)
    print(config["generator"])
    generator = Generator(config)
    
    size = (5,1,256,256)
    image = np.random.randint(0, 1, size)
    image = torch.from_numpy(image).float()
    features = generator(image)
    assert features.shape == size
    print(features.shape)