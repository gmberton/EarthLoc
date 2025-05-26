
import torch
import logging
from pathlib import Path

import datasets.utils as utils


def filter_images_far_from_POI(paths, point_of_interest, dist_threshold):
    coords = torch.stack([utils.get_nadir_center_from_path(p) for p in paths])
    distances_from_poi = utils.batch_geodesic_distances(point_of_interest, coords)
    assert len(distances_from_poi) == len(paths)
    # Select all images within dist_threshold km from POI
    paths = [p for p, d in zip(paths, distances_from_poi) if d < dist_threshold]
    return paths


class TestDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_path,
            dataset_name,
            db_paths,
            image_size,
            center_lat=45,
            center_lon=10,
            intersections_filename="queries_intersections_with_db_2021.pt",
            thresh_queries=2500,
            thresh_db=5000,
        ):
        self.dataset_name = dataset_name
        self.image_size = image_size
        
        point_of_interest = torch.tensor([[center_lat, center_lon]])
        q_db_intersections = torch.load(dataset_path / intersections_filename, weights_only=False)
        q_db_intersections = {q_path: [i[0] for i in inters] for q_path, inters in q_db_intersections.items()}

        # Get queries paths
        queries_paths = [p for p in q_db_intersections.keys()]
        # Remove the far away ones
        queries_paths = filter_images_far_from_POI(queries_paths, point_of_interest, thresh_queries)

        # Keep only DB images from 2021
        db_paths = [p for p in db_paths if utils.get_year_from_path(p) == 2021]
        # Remove the far away ones
        db_paths = filter_images_far_from_POI(db_paths, point_of_interest, thresh_db)
        db_paths = [Path(p) for p in db_paths]
        
        self.positives_per_query = []
        dict__db_path__idx = {path: i for i, path in enumerate(db_paths)}
        queries_paths_with_positives = []
        for q_path in queries_paths:
            positives = q_db_intersections[str(q_path)]
            if len(positives) == 0:
                logging.debug(f"Query {q_path} has no positives, it is probably over the sea. This query will be ignored.")
                continue
            
            positives = [dataset_path / p for p in positives]
            for pos_path in positives:
                if pos_path not in dict__db_path__idx:
                    logging.debug(f"Query {q_path} has positive {pos_path} which is not within dict__db_path__idx. "
                                  f"This could be a label error. This query will be ignored.")
                    break
            else:
                positives_indexes = [dict__db_path__idx[pos_path] for pos_path in positives]
                self.positives_per_query.append(positives_indexes)
                queries_paths_with_positives.append(q_path)
        
        logging.debug(f"TestDataset has {len(db_paths)} DB and {len(queries_paths)} queries")

        self.queries_paths = [dataset_path / "queries" / q_path for q_path in queries_paths_with_positives]
        self.db_paths = [db_path for db_path in db_paths]
        assert self.queries_paths[0].exists(), self.queries_paths[0]
        assert self.db_paths[0].exists(), self.db_paths[0]
        
        self.images_paths = self.db_paths + self.queries_paths
        self.num_db = len(self.db_paths)
        self.num_q = len(self.queries_paths)
    
    def __getitem__(self, index):
        path = self.images_paths[index]
        image = utils.load_image(path, self.image_size)
        return image, index, str(path)
    
    def __len__(self):
        return len(self.images_paths)
    
    def get_positives(self):
        return self.positives_per_query
    
    def __repr__(self):
        return f"< {self.dataset_name: <15} - #q: {self.num_q}; #db: {self.num_db} >"
