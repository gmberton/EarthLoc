
import torch
import random
import logging
import numpy as np
from PIL import Image
import torchvision.transforms as tfm

import datasets.utils as utils


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, clustered_paths, batch_size=32, size_before_transf=800):
        self.batch_size = batch_size
        self.size_before_transf = size_before_transf
        self.clustered_paths = clustered_paths
        
        num_all_paths = len([p for paths in clustered_paths for p in paths])
        logging.debug(f"TrainDataset has {num_all_paths} * 4 images")
    
    def __getitem__(self, index):
        paths = random.choice(self.clustered_paths)
        
        def are_tiles_overlapping(path1, path2):
            min_lat1, min_lon1, _, _, max_lat1, max_lon1, _, _, = utils.get_footprint_from_path(path1)
            min_lat2, min_lon2, _, _, max_lat2, max_lon2, _, _, = utils.get_footprint_from_path(path2)
            if max_lat1 <= min_lat2 or max_lat2 <= min_lat1: return False
            if max_lon1 <= min_lon2 or max_lon2 <= min_lon1: return False
            return True
        
        if len(paths) >= self.batch_size:
            chosen_paths = np.random.choice(paths, self.batch_size, replace=False).tolist()
        else:
            chosen_paths = np.random.choice(paths, self.batch_size, replace=True).tolist()
        
        images = torch.zeros([self.batch_size, 4, 3, self.size_before_transf, self.size_before_transf], dtype=torch.float32)
        for i1, path in enumerate(chosen_paths):
            every_year_paths = [utils.replace_year_in_path(path, 2021, y) for y in range(2018, 2022)]
            
            all_years_pil_imgs = [Image.open(p) for p in every_year_paths]
            for i2, pil_img in enumerate(all_years_pil_imgs):
                # Do not apply normalization here, it is applied later on GPU
                images[i1, i2] = tfm.ToTensor()(tfm.Resize(self.size_before_transf)(pil_img))
        
        is_overlapping = np.zeros([len(chosen_paths), len(chosen_paths)], dtype=np.int8)
        for i1, p1 in enumerate(chosen_paths):
            for i2, p2 in enumerate(chosen_paths):
                if are_tiles_overlapping(p1, p2):
                    is_overlapping[i1, i2] = 1
        
        chosen_paths = [str(p) for p in chosen_paths]
        return images, is_overlapping, chosen_paths
    
    def __len__(self):
        return 100_000  # Any random big number
    
    @staticmethod
    def create_random_clusters(paths, num_clusters):
        tmp_list = paths.copy()
        random.shuffle(tmp_list)
        clustered_paths = np.array_split(tmp_list, num_clusters)
        return clustered_paths
