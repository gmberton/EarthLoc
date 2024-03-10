
import sys
import torch
import logging
from pathlib import Path
from datetime import datetime

import test
import parser
import commons
from apl_models.apl_model import APLModel
from datasets.test_dataset import TestDataset


def eval_on_all_test_sets(model, dataset_path, db_paths_from_2021, log_dir,
                          num_preds_to_save=0, device="cuda"):
    for test_set_name, center_lat, center_lon in [
        ("Alps", 45, 10),
        ("Texas", 30, -95),
        ("Toshka Lakes", 23, 30),
        ("Amazon", -3, -60),
        ("Napa", 38, -122),
        ("Gobi", 40, 105),
    ]:
        test_dataset = TestDataset(
            dataset_path=dataset_path,
            dataset_name=test_set_name,
            db_paths=db_paths_from_2021,
            image_size=model.image_size,
            center_lat=center_lat,
            center_lon=center_lon,
        )
        recalls, recalls_str = test.test(test_dataset, model, log_dir,
                                         num_preds_to_save=num_preds_to_save, device=device)
        logging.info(f"Recalls on {test_set_name: <15} {test_dataset}: {recalls_str}")


if __name__ == "__main__":
    args = parser.parse_arguments()
    start_time = datetime.now()
    args.log_dir = Path("logs") / args.log_dir / start_time.strftime('%Y-%m-%d_%H-%M-%S')
    commons.make_deterministic(args.seed)
    commons.setup_logging(args.log_dir, console="debug")
    logging.info(" ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.log_dir}")
    
    model = APLModel()
    model = model.cuda()
    
    #### DATA ####
    db_paths_from_2021 = list((args.dataset_path / "database").glob("*/*/*.jpg"))
    
    if args.resume_model is not None:
        logging.info(f"Loading model from {args.resume_model}")
        model_state_dict = torch.load(args.resume_model)
        model.load_state_dict(model_state_dict)
    else:
        logging.info("WARNING: You didn't provide a path to resume the model (--resume_model parameter). " +
                    "Evaluation will be computed using randomly initialized weights.")
    eval_on_all_test_sets(model, args.dataset_path, db_paths_from_2021, args.log_dir,
                          num_preds_to_save=args.num_preds_to_save, device=args.device)
