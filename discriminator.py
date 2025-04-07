import torch.nn as nn
import torch.nn.functional as F
import yaml
import torch
from utils import load_config

class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.features = config["discriminator"]["features"]
        self.kernel_size = config["discriminator"]["kernel_size"]
        self.padding = config["discriminator"]["padding"]
        self.stride = config["discriminator"]["stride"]
        
        self.disc_blocks = nn.ModuleList()
        #[1,64,128,256,512,1]
        for i in range(len(self.features)-1):
            layers = []
            layers.append(nn.Conv2d(self.features[i], self.features[i+1], self.kernel_size[i], self.stride[i], self.padding[i]))
            if i!=0:
                layers.append(nn.BatchNorm2d(self.features[i+1]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            self.disc_blocks.append(nn.Sequential(*layers))
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        for block in self.disc_blocks:
            x = block(x)
        return self.sigmoid(x)
    
    
if __name__ == "__main__":
    config = load_config("config.yaml")
    discriminator = Discriminator(config)
    x = torch.randn(1, 1, 256, 256)
    print(discriminator(x).shape)