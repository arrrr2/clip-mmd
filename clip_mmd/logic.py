from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import clip_mmd.data as data
import clip_mmd.calculation as calculation
import numpy as np
import os.path as osp
from tqdm import tqdm
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class extractor():
    def __init__(self,
                 model_name="openai/clip-vit-large-patch14-336",
                 device="cuda", 
                 data_parallel=False,
                 device_ids=[0]):
        self.device = device
        self.data_parallel = data_parallel
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(device)
        if data_parallel:
            
            self.model = torch.nn.DataParallel(self.model.to(torch.device('cuda', device_ids[0])), device_ids=device_ids)
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.processor.do_rescale=False
        self.processor.do_center_crop=False
        self.processor.do_normalize=True
        self.processor.do_resize=False
        self.processor.do_convert_rgb=False
        self.holder = torch.device('cuda', device_ids[0]) if data_parallel else torch.device('cuda')
        self.model.eval()
        

    @torch.no_grad()
    def __call__(self, img:torch.Tensor):
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.holder) for k, v in inputs.items()}
        features = self.model(**inputs).image_embeds.cpu()
        features /= torch.linalg.norm(features, axis=-1, keepdims=True)
        return features


class CMMD():
    def __init__(self, infer=True,
                    extract_model="openai/clip-vit-large-patch14-336", img_size=(336, 336), 
                    device="cuda", data_parallel=False, device_ids=[0], interpolation="bicubic",
                    feat_bs=64, num_workers=8, compute_bs=256, low_mem=True,
                    original=False
                    ):
        self.extractor = extractor(extract_model, device, data_parallel, device_ids) if infer else None
        self.feat_bs = feat_bs
        self.img_size = img_size
        self.interpolation = interpolation
        self.compute_bs = compute_bs
        self.low_mem = low_mem
        self.num_workers = num_workers
        self.device=device
        self.original = original
        self.first_dev = 0 if device_ids is None else device_ids[0]
    def prepare_folder(self, data_folder:str):
        dataset = data.folder_dataset(data_folder, self.img_size, self.interpolation)
        loader = DataLoader(dataset, batch_size=self.feat_bs, num_workers=self.num_workers)
        return loader
    
    def prepare_pack(self, data_path:str):
        dataset = data.batched_dataset(data_path, self.img_size, self.interpolation)
        loader = DataLoader(dataset, batch_size=self.feat_bs, num_workers=self.num_workers)
        return loader
    
    def prepare_custom_dataset(self, custom_dataset):
        loader = DataLoader(custom_dataset, batch_size=self.feat_bs, num_workers=self.num_workers)
        return loader
        
    def calculate_statics(self, loader):
        features = []
        for batch in tqdm(loader, leave=False):
            features.append(self.extractor(batch).cpu())
        features = torch.cat(features, dim=0)
        return features
    
    def calculate_mmd(self, x, y):
        if self.original:
            return calculation.mmd_efficient(x, y)
        else:
            return calculation.mmd_efficient(x, y, low_mem=self.low_mem, coeff_bs=self.compute_bs, device=self.device)
    
    def get_prepared_statics(self, data_path:str):
        return data.get_stastics(data_path)

    def parse_file(self, input):
        if isinstance(input, str):
            if osp.isdir(input):
                return input, "folder"
            elif input.endswith(('.npz', 'npy')):
                data = np.load(input)
                torch_data = torch.from_numpy(data)
            elif input.endswith(('.pt', '.pth', '.pkl')):
                torch_data = torch.load(input)
        elif isinstance(input, np.ndarray):
            torch_data = torch.from_numpy(input)
        elif not isinstance(input, torch.Tensor): 
            raise TypeError("Not recognized data content")
        else: torch_data = input
        data_shape = torch_data.shape
        if len(data_shape) == 2:
            return torch_data, "stastics"
        elif len(data_shape) == 4:
            return torch_data, "pack"
        raise ValueError("Invalid file format")
    
    def prepare_input(self, x):
        x_computed = False
        if isinstance(x, Dataset):
            x = self.prepare_custom_dataset(x)
        else:
            x, x_type = self.parse_file(x)
            if x_type == "pack":
                x = self.prepare_pack(x)
            elif x_type == "folder":
                x = self.prepare_folder(x)
            elif x_type == "stastics":
                x_computed = True
            else:
                raise ValueError("Invalid type")
        return x, x_computed
    
    def execute(self, x, y):
        assert self.extractor is not None, "Extractor is not initialized"
        x, x_computed = self.prepare_input(x)
        y, y_computed = self.prepare_input(y)
        if not x_computed:
            x = self.calculate_statics(x)
        if not y_computed:
            y = self.calculate_statics(y)
        return self.calculate_mmd(x, y)
        


    