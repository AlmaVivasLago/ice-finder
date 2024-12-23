import os
import cv2
import torch
from torch.utils.data import Dataset
from glob import glob

class DatasetInference(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        ups = os.path.join(root_dir, '*.tiff')
        self.file_paths = glob(os.path.join(root_dir, '*.tiff'))  # Load all .tif files

    def _load_image(self, fpath):
        x = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        x = x / 255.
        return (1 - x).unsqueeze(0)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        sample = self._load_image(img_path)
                
        if self.transform:
            sample = self.transform(sample)
        
        return sample


