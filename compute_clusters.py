
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm

import datasets.utils as utils
from datasets.base_dataset import BaseDataset


def compute_clusters(model, all_paths, num_clusters=100,
                     device="cuda", num_workers=8, batch_size=32):
    
    # Keep only DB images from 2021
    all_paths = [p for p in all_paths if utils.get_year_from_path(p) == 2021]
    test_dataset = BaseDataset(all_paths, model.image_size)
    
    dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset, num_workers=num_workers, batch_size=batch_size
    )
    model = model.eval()
    all_descs = np.empty((len(test_dataset), model.desc_dim), dtype=np.float32)
    
    with torch.inference_mode():
        for images, indices, _ in tqdm(dataloader, ncols=120,
                                    desc="Computing descriptors for clustering"):
            descriptors = model(images.to(device))
            descriptors = descriptors.cpu().numpy()
            all_descs[indices.numpy(), :] = descriptors
    
    logging.debug(f"Start computing clusters with {len(all_paths)} paths")
    
    
    kmeans = faiss.Kmeans(model.desc_dim, num_clusters, niter=100, verbose=True)
    kmeans.train(all_descs)
    cluster_ids_x = kmeans.index.search(all_descs, 1)[1][:, 0]
    
    clustered_paths = [[] for _ in range(num_clusters)]
    for cl_id, path in zip(cluster_ids_x, all_paths):
        clustered_paths[int(cl_id)].append(path)
    
    return clustered_paths
