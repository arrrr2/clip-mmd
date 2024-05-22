import torch
from PIL import Image
from torch.utils.data import Dataset
import glob
import os.path as osp
import numpy as np
from typing import Tuple

def resizer(img:Image.Image, size: Tuple[int, int], interpolation="bicubic"):
    resamplers = {
        'nearest': Image.Resampling.NEAREST,
        'bilinear': Image.Resampling.BILINEAR,
        'bicubic': Image.Resampling.BICUBIC,
        'lanczos': Image.Resampling.LANCZOS,
        'box': Image.Resampling.BOX,
        'hamming': Image.Resampling.HAMMING,
    }
    h_hat, w_hat = size
    h, w = img.size
    if h == size[0] and w == size[1]:
        return img
    if (h/w) > (h_hat/w_hat):
        w_new = w_hat
        h_new = int(h * w_hat / w)
    else:
        h_new = h_hat
        w_new = int(w * h_hat / h)
    img = img.resize((w_new, h_new), resample=resamplers[interpolation]) 
    left = (w_new - w_hat) // 2
    top = (h_new - h_hat) // 2
    right = (w_new + w_hat) // 2
    bottom = (h_new + h_hat) // 2
    img = img.crop((left, top, right, bottom))
    return img

class folder_dataset(Dataset):
    img_ext = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif', 'webp', 'avif', 'heic', 'jxl']
    def __init__(self, folder:str, size:Tuple[int, int], interpolation="bicubic"):
        
        all_files = glob.glob(osp.join(folder, "**", "*"), recursive=True)
        self.files = [file for file in all_files if osp.splitext(file)[1][1:].lower() in self.img_ext]
        
        self.size = size
        self.interpolation = interpolation
        self.size = size
        self.interpolation = interpolation

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        img = resizer(img, self.size, self.interpolation)
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255
        return img

class batched_dataset(Dataset):
    def __init__(self, data, size:Tuple[int, int], interpolation="bicubic"):
        self.data = data
        self.size = size
        self.interpolation = interpolation
        if self.data.shape[-1] == 3:
            self.data = self.data.permute(0, 3, 1, 2)
        self.pix_range = (self.data.min(), self.data.max())
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img: torch.Tensor = self.data[idx]
        if img.dtype == torch.uint8:
            img = img.permute(1, 2, 0).numpy()
        elif img.dtype == torch.float32:
            img = img.permute(1, 2, 0).numpy()
            if self.pix_range[0] < 1.01:
                img = (img * 255 + 0.5).round().to(torch.uint8).permute(1,2,0).numpy()
        img = resizer(Image.fromarray(img), self.size, self.interpolation)
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255
        return img
        
def get_stastics(path: str):
    return torch.load(path)

