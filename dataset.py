import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt

class SheetletCellDataset(Dataset):
    
    def __init__(self, path_to_npy, transform=None):
        super().__init__()
        self.path_to_npy = path_to_npy
        self.transform = transform
        self.data = np.load(path_to_npy,allow_pickle=True)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        images = self.data[idx]
        images = torch.tensor(images)
        if self.transform:
            images = self.transform(images)
        return images[0,:,:], images[1,:,:]


if __name__ == "__main__":
    
    transform = transforms.Compose([
        transforms.Pad(padding=20,padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10,fill=0),
        transforms.RandomAffine(0, translate=(0.05,0.05), scale=(0.9,1.15), shear=2),
        #transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        #transforms.CenterCrop(size=(512,512))        
    ])
    
    not_transformed_dataset = SheetletCellDataset(path_to_npy="combined_patches.npy")
    traindataset = SheetletCellDataset(path_to_npy="combined_patches.npy", transform=transform)
    
    condition, target = traindataset[0]
    not_transformed_condition, not_transformed_target = not_transformed_dataset[0]
    print(condition.shape)
    print(target.shape)
    print(not_transformed_condition.shape)
    print(not_transformed_target.shape)
    
    fig, axes = plt.subplots(2,2)
    axes[0,0].imshow(not_transformed_condition)
    axes[0,1].imshow(not_transformed_target)
    axes[1,0].imshow(condition)
    axes[1,1].imshow(target)
    plt.savefig("transformed_images.png")
    
   
        
        

        
        
    