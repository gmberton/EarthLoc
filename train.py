
import os
import sys
import torch
import einops
import logging
import torchmetrics
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity

import eval
import test
import parser
import commons
import augmentations
import compute_clusters
from apl_models.apl_model import APLModel
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset

args = parser.parse_arguments()
start_time = datetime.now()
args.log_dir = Path("logs") / args.log_dir / start_time.strftime('%Y-%m-%d_%H-%M-%S')
commons.make_deterministic(args.seed)
commons.setup_logging(args.log_dir, console="debug")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.log_dir}")

model = APLModel()
model = model.to(args.device)

#### DATA ####
db_paths = list((args.dataset_path / "database").glob("*/*/*.jpg"))

randomly_clustered_paths = TrainDataset.create_random_clusters(db_paths, args.num_clusters)
train_dataset = TrainDataset(
    clustered_paths=randomly_clustered_paths,
    batch_size=args.batch_size,
    size_before_transf=args.size_before_transf,
)

val_dataset = TestDataset(
    dataset_path=args.dataset_path,
    dataset_name="val",
    db_paths=db_paths,
    image_size=model.image_size,
    center_lat=30,
    center_lon=-95,
    thresh_queries=500,
    thresh_db=1200,
)

dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1, num_workers=args.num_workers, shuffle=True
)

augmentation = augmentations.get_my_augment(
    distortion_scale=args.dist_scale, crop_size=args.crop_size, final_size=model.image_size,
    rand_rot=args.rand_rot, brightness=args.brightness, contrast=args.contrast, saturation=args.saturation, hue=args.hue
)

#### LOSSES & OPTIM ####
criterion = losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity())
miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=CosineSimilarity())
optim = torch.optim.Adam(model.parameters(), lr=args.lr)

best_r5 = not_improved_num = 0

for num_epoch in range(args.num_epochs):
    
    if num_epoch != 0 and num_epoch % args.compute_clusters_every_n_epochs == 0:
        clustered_paths = compute_clusters.compute_clusters(
            model, all_paths=db_paths,
            num_clusters=args.num_clusters,
            device=args.device, batch_size=args.batch_size, num_workers=args.num_workers
        )
        train_dataset = TrainDataset(
            clustered_paths=clustered_paths,
            batch_size=args.batch_size,
            size_before_transf=args.size_before_transf,
        )
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                                  num_workers=args.num_workers,
                                                  drop_last=True, shuffle=True)
    
    model = model.train()
    mean_loss = torchmetrics.MeanMetric()
    mean_batch_acc = torchmetrics.MeanMetric()
    tqdm_bar = tqdm(dataloader, total=args.iterations_per_epoch, ncols=120)
    for iteration, (images, is_overlapping, chosen_paths) in enumerate(tqdm_bar):
        if iteration >= args.iterations_per_epoch:
            break
        with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
            images = einops.rearrange(images, "one bs years c h w -> (one bs) years c h w",
                                      bs=args.batch_size, one=1, years=4)
            images = images.to(args.device)
            
            # Apply same augmentation to images from same year, i.e. Year-Wise Augmentation
            views = [augmentation(images[:, year]) for year in range(4)]
            
            views = einops.rearrange(views, "nv b c h w -> (b nv) c h w", nv=4, b=args.batch_size)
            
            descriptors = model(views)
            labels = torch.repeat_interleave(torch.arange(args.batch_size), 4)
            miner_outputs = miner(descriptors, labels)
            
            # Filter away overlapping pairs of images, i.e. Neutral-Aware MS loss
            anchors, negatives = miner_outputs[2:]
            is_non_overlapping = is_overlapping.to(args.device)[0, anchors//4, negatives//4] == 0
            far_indexes = torch.where(is_non_overlapping)[0]
            anchors = anchors[far_indexes]
            negatives = negatives[far_indexes]
            miner_outputs = tuple([miner_outputs[0], miner_outputs[1], anchors, negatives])
            
            loss = criterion(descriptors, labels, miner_outputs)
        loss.backward()
        
        # calculate the % of trivial pairs/triplets which do not contribute in the loss value
        nb_samples = descriptors.shape[0]
        nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
        batch_acc = (1.0 - (nb_mined / nb_samples)) * 100
        
        optim.step()
        optim.zero_grad()
        mean_loss.update(loss.item())
        mean_batch_acc.update(batch_acc)
        tqdm_bar.desc = f"Loss: {mean_loss.compute()} - batch_acc: {batch_acc:.1f} - {nb_samples} - {nb_mined}"
    
    recalls, recalls_str = test.test(val_dataset, model, device=args.device)

    r5 = recalls[1]
    logging.debug(f"Recalls: {recalls_str}")
    
    is_best = r5 > best_r5
    if is_best:
        prev_best_model = list(args.log_dir.glob("best_*"))
        if len(prev_best_model) != 0:  # Delete previous best_model file
            os.remove(prev_best_model[0])
        torch.save(model.state_dict(), args.log_dir / f"best_model_{r5:.1f}.pt")
        prev_ckpt = list(args.log_dir.glob("ckpt_*"))
        if len(prev_ckpt) != 0:  # Delete previous ckpt file
            os.remove(prev_ckpt[0])
        torch.save({
            "epoch_num": num_epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "best_r5": best_r5
        }, args.log_dir / f"ckpt_e{num_epoch:02d}_{r5:.1f}.pt")
            
        logging.debug(f"Improved: previous best r5 = {best_r5:.1f}, current r5 = {r5:.1f}")
        best_r5 = r5
        not_improved_num = 0
    else:
        not_improved_num += 1
        logging.debug(f"Not improved: {not_improved_num} / {args.patience}: best r5 = {best_r5:.1f}, current r5 = {r5:.1f}")
    
    logging.info(f"Epoch {num_epoch: >2} - loss: {mean_loss.compute():.2f} - "
                 f"mean batch_acc: {mean_batch_acc.compute():.1f} - "
                 f"patience left: {args.patience - not_improved_num} - best r5: {best_r5:.1f} - {recalls_str[:20]}")
    
    if not_improved_num >= args.patience:
        logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training after {str(datetime.now() - start_time)[:-7]}.")
        break

logging.info(f"Training finished in {str(datetime.now() - start_time)[:-7]}")

logging.debug("Testing with the best model")

best_model_path = list(args.log_dir.glob("best_*"))[0]
best_model_state_dict = torch.load(best_model_path, weights_only=True)
model.load_state_dict(best_model_state_dict)

eval.eval_on_all_test_sets(model, args.dataset_path, db_paths, args.log_dir,
                           num_preds_to_save=args.num_preds_to_save, device=args.device)
