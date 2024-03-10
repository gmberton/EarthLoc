
import torch

import datasets.utils as utils


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, images_paths, image_size):
        self.images_paths = images_paths
        self.image_size = image_size
    
    def __getitem__(self, index):
        path = self.images_paths[index]
        image = utils.load_image(path, self.image_size)
        return image, index, str(path)
    
    def __len__(self):
        return len(self.images_paths)
