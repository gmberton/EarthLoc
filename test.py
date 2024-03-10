
import faiss
import torch
import einops
import logging
import numpy as np
from tqdm import tqdm
import torchvision.transforms as tfm
from torch.utils.data import DataLoader

import visualizations


# Compute R@1, R@5, R@10, R@20, R@100
RECALL_VALUES = [1, 5, 10, 20, 100]


def test(eval_ds, model, log_dir=None, num_preds_to_save=0, device="cuda", batch_size=16, num_workers=4):
    model = model.eval()
    logging.debug("Test - computing descriptors")
    all_descriptors = np.empty((4, len(eval_ds), model.desc_dim), dtype="float32")
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            dataloader = DataLoader(dataset=eval_ds, num_workers=num_workers,
                                    batch_size=batch_size)
            for images, indices, _ in tqdm(dataloader, ncols=120, desc="Computing descriptors"):
                for rot_idx in range(4):
                    angle = [0, 90, 180, 270][rot_idx]
                    images = images.to(device)
                    rot_images = tfm.functional.rotate(images, angle)
                    descriptors = model(rot_images)
                    descriptors = descriptors.cpu().numpy()
                    all_descriptors[rot_idx, indices.numpy(), :] = descriptors
    
    q_descriptors = all_descriptors[0, eval_ds.num_db:]  # Only non-rotated queries
    db_descriptors = all_descriptors[:, :eval_ds.num_db]  # 4 rotation db images
    db_descriptors = einops.rearrange(db_descriptors, "four db dim -> (four db) dim")
    
    logging.debug("Test - computing kNN and predictions")
    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(model.desc_dim)
    faiss_index.add(db_descriptors)
    
    _, predictions = faiss_index.search(q_descriptors, max(RECALL_VALUES))
    preds_angles = predictions // eval_ds.num_db * 90
    predictions = predictions % eval_ds.num_db
    
    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    recalls = np.zeros(len(RECALL_VALUES))
    for query_index, preds in enumerate(predictions):
        for i, n in enumerate(RECALL_VALUES):
            if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by num_q and multiply by 100, so the recalls are in percentages
    recalls = recalls / eval_ds.num_q * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls)])
    
    #### From here till end of function it is only for visualizations
    if num_preds_to_save != 0:
        logging.debug("Test - computing visualizations")
        visualizations.compute_visualizations(eval_ds, log_dir, num_preds_to_save,
                                              predictions, preds_angles, positives_per_query)
    
    return recalls, recalls_str

