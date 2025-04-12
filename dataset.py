import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import matplotlib.pyplot as plt

class SheetletCellDataset(Dataset):
    
    def __init__(self, data, transform=None):
        """
        data: list of tuples (condition, target)
        """
        super().__init__()
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Convert to tensor and add channel dimension
        condition = torch.from_numpy(self.data[idx][0]).unsqueeze(0)  # Add channel dim 1 512 512
        target = torch.from_numpy(self.data[idx][1]).unsqueeze(0)     # Add channel dim 1 512 512
        stacked = torch.cat([condition, target], dim=0) 

        if self.transform is not None:
            # Apply the same random seed for both images
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            stacked = self.transform(stacked)

        condition, target = stacked[0].unsqueeze(0), stacked[1].unsqueeze(0) # 1 512 512
        return condition, target


        
        
    