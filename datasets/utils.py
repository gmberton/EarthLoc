
import torch
from PIL import Image
from pathlib import Path
import torchvision.transforms as tfm


def get_image_metadata_from_path(path):
    if isinstance(path, Path):
        return path.name.split("@")
    elif isinstance(path, str):
        return path.split("@")


def replace_year_in_path(path, src_year, dst_year):
    new_path = str(path).replace(f"/{src_year}_", f"/{dst_year}_")
    new_path = new_path.replace(f"@{src_year}@", f"@{dst_year}@")
    return Path(new_path)


def get_year_from_path(path):
    splits = get_image_metadata_from_path(path)
    year = int(splits[-6][:4])
    assert 1900 < year < 2030
    return year


def get_orientation_from_path(path):
    splits = get_image_metadata_from_path(path)
    return int(float(splits[-2]))


def get_footprint_area_from_path(path):
    splits = get_image_metadata_from_path(path)
    return int(splits[13])


def get_footprint_from_path(path):
    splits = get_image_metadata_from_path(path)
    coords = splits[1:9]
    coords = [float(c) for c in coords]
    lat1, lon1, lat2, lon2, lat3, lon3, lat4, lon4 = coords
    return lat1, lon1, lat2, lon2, lat3, lon3, lat4, lon4


def get_nadir_center_from_path(path):
    splits = get_image_metadata_from_path(path)
    nadir_lat, nadir_lon = splits[11:13]
    nadir_lat = float(nadir_lat)
    nadir_lon = float(nadir_lon)
    return torch.tensor([nadir_lat, nadir_lon])


def batch_geodesic_distances(origin, destination):
    assert type(origin) == type(destination) == torch.Tensor
    assert origin.shape[1] == destination.shape[1] == 2
    radius = 6371 # km
    lat1, lon1 = origin.T
    lat2, lon2 = destination.T
    dlat = torch.deg2rad(lat2-lat1)
    dlon = torch.deg2rad(lon2-lon1)
    a = torch.sin(dlat/2) * torch.sin(dlat/2) + torch.cos(torch.deg2rad(lat1)) \
        * torch.cos(torch.deg2rad(lat2)) * torch.sin(dlon/2) * torch.sin(dlon/2)
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
    distances = radius * c
    return distances


def load_image(image_path, image_size):
    pil_image = Image.open(image_path)
    transform = tfm.Compose([
        tfm.Resize(image_size, antialias=True),
        tfm.ToTensor(),
        tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(pil_image)
