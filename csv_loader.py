import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import tifffile as tiff
from PIL import Image
from torchvision import datasets, transforms, models

class load_csv(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = torch.from_numpy(tiff.imread(img_path)).permute(2,0,1).float()
        #Image.MAX_IMAGE_PIXELS = None
                
        '''
        train_transform = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.ToPILImage(),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])
        '''
        image.transform = transforms.RandomResizedCrop(224)
        
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        
        #if self.transform:
        #    image = self.transform(image)
        
        return (image, y_label)

    