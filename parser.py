import argparse
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=96, help="_")
    parser.add_argument("--patience", type=int, default=10, help="_")
    parser.add_argument("--iterations_per_epoch", type=int, default=500, help="_")
    parser.add_argument("--num_epochs", type=int, default=20, help="_")
    parser.add_argument("--lr", type=float, default=0.0001, help="_")

    parser.add_argument("--compute_clusters_every_n_epochs", type=int, default=4, help="_")
    parser.add_argument("--num_clusters", type=int, default=50, help="_")
    
    # Data augmentation
    parser.add_argument("--size_before_transf", type=int, default=800,
                        help="image size before applying augmentations")
    parser.add_argument("--crop_size", type=int, default=700,
                        help="size of random crop")
    parser.add_argument("--image_size", type=int, default=320,
                        help="image size for train and test")
    parser.add_argument("--rand_rot", type=int, default=45,
                        help="random rotation augmentation")
    parser.add_argument("--dist_scale", type=float, default=0.5,
                        help="distortion augmentation")
    parser.add_argument("--brightness", type=float, default=0.9,
                        help="color jittering")
    parser.add_argument("--contrast", type=float, default=0.9,
                        help="color jittering")
    parser.add_argument("--saturation", type=float, default=0.9,
                        help="color jittering")
    parser.add_argument("--hue", type=float, default=0.0,
                        help="color jittering")
    
    # Others
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="_")
    parser.add_argument("--seed", type=int, default=0, help="_")
    parser.add_argument("--num_workers", type=int, default=3, help="_")
    
    parser.add_argument("--resume_model", type=str, default=None,
                        help="pass the path of a best_model.torch file to load its weights")
    
    # Visualizations
    parser.add_argument("--num_preds_to_save", type=int, default=20,
                        help="Save visualizations of N queries and their predictions")
    
    # Paths
    parser.add_argument("--dataset_path", type=str, default="./data", help="_")
    parser.add_argument("--log_dir", type=str, default="default",
                        help="name of directory on which to save the logs, under logs/log_dir")
    
    args = parser.parse_args()
    args.dataset_path = Path(args.dataset_path)
    
    return args

