import torch 
import torch.nn as nn
import torch.optim as optim
import yaml
import numpy as np

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
        x, skip_connections = self.encoder(x, z)
        x = self.decoder(x, skip_connections)
        return x
    
class Encoder(nn.Module):
    def __init__(self, in_channel, features, kernel_size, stride, noise_dim, padding):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        self.features = features
        self.noise_dim = noise_dim
        #1-64 block, 64-128 block, 128-256 block, 256-512 block
        for i in range(len(features)):
            layers = []
            layers.append(nn.Conv2d(in_channel, features[i], kernel_size, stride, padding))
            #Only to first block
            if i > 0:
                layers.append(nn.BatchNorm2d(features[i]))
            layers.append(nn.LeakyReLU(0.2))
            in_channel = features[i]
            self.encoder_blocks.append(nn.Sequential(*layers))
        self.noise_processor = nn.Linear(noise_dim, self.features[0]//2*self.features[0]//2*noise_dim)
        
    def forward(self, x, z):
        skip_connections = []
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            skip_connections.append(x)
        z = self.noise_processor(z)
        z = z.view(z.size(0), self.noise_dim, self.features[0]//2, self.features[0]//2)
        x = torch.cat([x, z], dim=1)
        return x, skip_connections


class Decoder(nn.Module):
    def __init__(self, input_features, features, kernel_size, stride,noise_dim, padding):
        super().__init__()
        self.decoder_blocks = nn.ModuleList()
        #512+noise_dim - 256 block, 
        #512-256 block, 256-128 block, 128-64 block
        for i in range (len(features)-1,0,-1):
            layers = []
            if i==len(features)-1:
                layers.append(nn.ConvTranspose2d(features[i]+noise_dim, features[i-1], kernel_size, stride, padding))
            else:
                layers.append(nn.ConvTranspose2d(features[i]*2, features[i-1], kernel_size, stride, padding))                
            layers.append(nn.BatchNorm2d(features[i-1]))
            layers.append(nn.ReLU())
            self.decoder_blocks.append(nn.Sequential(*layers))
        
        self.decoder_blocks.append(nn.Sequential(
            nn.ConvTranspose2d(features[0]*2, input_features, kernel_size, stride, padding),
            nn.Tanh()
        ))
        
        #512+noise_dim - 256 block, 256*2 - 128 block, 128*2 - 64 block, 64*2 - 1 block
        #encoder 1-64 , 64-128, 128-256, 256-512
        #skip connections: 64, 128, 256, 512

    def forward(self, x, skip_connections):
        skip_connections = skip_connections[::-1]
        x = self.decoder_blocks[0](x)
        for i in range(1,len(self.decoder_blocks)):
            x = torch.cat([x, skip_connections[i]], dim=1)
            x = self.decoder_blocks[i](x)
        return x
        
if __name__ == "__main__":
    
    def load_config(path):
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return config
    
    config = load_config("config.yaml")
    print(config)
    print(config["generator"])
    generator = Generator(config)
    
    image = np.random.randint(0, 255, (1, 1, 512, 512))
    z = np.random.randint(0, 100, (1, 128))
    image = torch.from_numpy(image).float()
    z = torch.from_numpy(z).float()
    features = generator(image,z)
    print(features.shape)