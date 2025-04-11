import torch 
import torch.nn as nn
import torch.optim as optim
import yaml
import numpy as np
from utils import load_config
import torch.nn.functional as F

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
        
    def forward(self, x, z):
        # Store the mask of white regions for attention
        white_mask = (x > 0.8).float()  # Assuming white is close to 1.0
        x, skip_connections = self.encoder(x, z)
        x = self.decoder(x, skip_connections, white_mask)
        return x
    
class Encoder(nn.Module):
    def __init__(self, in_channel, features, kernel_size, stride, noise_dim, padding):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        self.features = features
        self.noise_dim = noise_dim
        self.noise_injection = NoiseInjection()
        
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
        
        # Use more noise channels in latent space
        self.noise_processor = nn.Sequential(
            nn.Linear(noise_dim, self.features[0]//2*self.features[0]//2*noise_dim)
        )
        
    def forward(self, x, z):
        skip_connections = []
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            # Add some noise at each level to encourage diversity
            #if i > 0:
                #x = self.noise_injection(x)
            skip_connections.append(x)
        
        # Process noise and make it more impactful
        z = self.noise_processor(z)
        z = z.view(z.size(0), self.noise_dim, self.features[0]//2, self.features[0]//2)
        x = torch.cat([x, z], dim=1)
        return x, skip_connections


class Decoder(nn.Module):
    def __init__(self, input_features, features, kernel_size, stride, noise_dim, padding):
        super().__init__()
        self.decoder_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        self.noise_dim = noise_dim
        self.noise_injection = NoiseInjection()
        
        # 512+noise_dim - 256 block, 256-128 block, 128-64 block, 64-1 block
        for i in range(len(features)-1, 0, -1):
            layers = []
            if i == len(features)-1:
                layers.append(nn.ConvTranspose2d(features[i]+noise_dim, features[i-1], kernel_size, stride, padding))
            else:
                layers.append(nn.ConvTranspose2d(features[i]*2, features[i-1], kernel_size, stride, padding))                
            layers.append(nn.BatchNorm2d(features[i-1]))
            layers.append(nn.ReLU())
            #layers.append(nn.Dropout(0.05))
            self.decoder_blocks.append(nn.Sequential(*layers))
            
            # Add attention blocks
            if i < len(features)-1:
                self.attention_blocks.append(SpatialAttention(features[i-1]))
        
        # Final layer
        self.decoder_blocks.append(nn.Sequential(
            nn.ConvTranspose2d(features[0]*2, input_features, kernel_size, stride, padding),
            nn.Tanh()
        ))
        
    def forward(self, x, skip_connections, white_mask=None):
        skip_connections = skip_connections[::-1]
        x = self.decoder_blocks[0](x)
        
        for i in range(1, len(self.decoder_blocks)):
            # Concat with skip connection
            x = torch.cat([x, skip_connections[i]], dim=1)
            # Apply decoder block
            x = self.decoder_blocks[i](x)  
            # Normalize to [0,1]
            x = (x+1)/2      
        return x


class NoiseInjection(nn.Module):
    def __init__(self, strength=0.05):
        super().__init__()
        self.strength = strength
        
    def forward(self, x):
        if not self.training:
            return x
            
        noise = torch.randn_like(x) * self.strength
        return x + noise


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # Simple self-attention mechanism
        attention = torch.sigmoid(self.conv(x))
        out = x * attention
        return x + self.gamma * out
        
if __name__ == "__main__":
    
    config = load_config("config.yaml")
    print(config)
    print(config["generator"])
    generator = Generator(config)
    
    size = (5,1,512,512)
    image = np.random.randint(0, 255, size)
    z = np.random.randint(0, 100, (5, config["generator"]["noise_dim"]))
    image = torch.from_numpy(image).float()
    z = torch.from_numpy(z).float()
    features = generator(image,z)
    assert features.shape == size
    print(features.shape)