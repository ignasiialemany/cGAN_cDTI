import torch 
import torch.nn as nn
import torch.optim as optim
import yaml
import numpy as np

class Encoder(nn.Module):
    def __init__(self, in_channel, features, kernel_size, stride, noise_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(features)):
            self.layers.append(nn.Conv2d(in_channel, features[i], kernel_size, stride))
            if i > 0:
                self.layers.append(nn.BatchNorm2d(features[i]))
            self.layers.append(nn.LeakyReLU(0.2))
            in_channel = features[i]
        self.noise_processor = nn.Linear(noise_dim, 30*30*128)
        
    def forward(self, x, z):
        skip_connections = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                skip_connections.append(x)
        z = self.noise_processor(z)
        z = z.view(z.size(0), 128, 30, 30)
        x = torch.cat([x, z], dim=1)
        return x, skip_connections

class Decoder(nn.Module):
    def __init__(self, features, kernel_size, stride, noise_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(features)):
            self.layers.append(nn.ConvTranspose2d(features[i], features[i-1], kernel_size, stride))
            
        



class SheetletToCellsGenerator(nn.Module):
    
    def __init__(self, config):
        
        super().__init__()
        self.config = config
        
        self.encoder = Encoder(config["encoder"]["in_channels"], config["encoder"]["features"], 
                               config["encoder"]["kernel_size"], config["encoder"]["stride"], 
                               config["encoder"]["noise_dim"])
        
        # self.noise_processor = nn.Linear(noise_dim, 25*25*128)
                
        # # Bottleneck (where features and noise combine)
        # self.bottleneck = nn.Sequential(
        #     nn.Conv2d(512+128, 512, 3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.LeakyReLU(0.2)
        # )
        
        # # Decoder (upsampling path)
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 25x25 → 50x50
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 50x50 → 100x100
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 100x100 → 200x200
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),  # 200x200 → 400x400
        #     nn.Tanh()  # Output in range [-1, 1]
        # )
    
    def forward(self, x):
        # Encode the binary mask
        features = self.encoder(x)
        return features
    
    

if __name__ == "__main__":
    
    def load_config(path):
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return config
    
    config = load_config("config.yaml")
    print(config)
    print(config["encoder"])
    generator = SheetletToCellsGenerator(config)
    
    image = np.random.randint(0, 255, (1, 1, 512, 512))
    image = torch.from_numpy(image).float()
    features = generator(image)
    print(features.shape)