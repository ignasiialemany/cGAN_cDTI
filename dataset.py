import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2

class CustomDataset(Dataset):
    