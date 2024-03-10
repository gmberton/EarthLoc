

import io
import os
import csv
import time
import imageio
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from multiprocessing import Pool
from urllib.request import Request, urlopen

NUM_PROCESSES = 16

BASE_URL = "https://s2maps-tiles.eu/wmts?layer=s2cloudless-__YEAR___3857&" \
    "style=default&tilematrixset=GoogleMapsCompatible&Service=WMTS" \
    "&Request=GetTile&Version=1.0.0&Format=image%2Fjpeg" \
    "&TileMatrix=__ZOOM__&TileCol=__COL__&TileRow=__ROW__"


def download_image(url, num_tries=4):
    """Return PIL Image from the URL"""
    for i in range(num_tries):  # Try the download num_tries
        try:
            req = Request(url=url, headers={'User-Agent': 'Mozilla/5.0'})
            req = urlopen(req, timeout=5)
            return Image.open(io.BytesIO(req.read())).convert("RGB")
        except Exception as e:
            if i == num_tries:
                raise e
            else:
                print(f"Couldn't download {url} due to {e}, will retry in 10 seconds")
                time.sleep(10)


def download_region(metadata):
    footprint, zoom, row, col, year, nadir_lat, nadir_lon, sq_km_area, orientation = read_metadata(metadata)
    for year in [2018, 2019, 2020, 2021]:
        folder_name = Path("data") / "database" / f"{year}_{zoom:02d}" / f"{int(round(nadir_lat, -1))}_{int(round(nadir_lon, -1))}"
        image_name = \
            f"@{footprint[0]:.5f}@{footprint[1]:.5f}@{footprint[2]:.5f}@{footprint[3]:.5f}" \
            f"@{footprint[4]:.5f}@{footprint[5]:.5f}@{footprint[6]:.5f}@{footprint[7]:.5f}" \
            f"@{zoom:02d}_{row:04d}_{col:04d}@{year}@{nadir_lat:.5f}@{nadir_lon:.5f}" \
            f"@{sq_km_area}@0@.jpg"
        image_path = folder_name / image_name
        if image_path.exists():
            continue
        image = np.zeros([1024, 1024, 3], dtype=np.uint8)
        for r in range(4):
            for c in range(4):
                url = BASE_URL\
                    .replace('__YEAR__', str(year)).replace('__ZOOM__', str(zoom))\
                    .replace('__ROW__', str(row+r)).replace('__COL__', str(col+c))
                tile = download_image(url)
                image[r*256 : (r+1)*256, c*256 : (c+1)*256] = np.array(tile)
        os.makedirs(folder_name, exist_ok=True)
        imageio.imsave(image_path, image)


def read_metadata(metadata):
    footprint = (
        float(metadata[0]), float(metadata[1]), float(metadata[2]), float(metadata[3]),
        float(metadata[4]), float(metadata[5]), float(metadata[6]), float(metadata[7]),
    )
    zoom = int(metadata[8])
    row = int(metadata[9])
    col = int(metadata[10])
    year = int(metadata[11])
    nadir_lat = float(metadata[12])
    nadir_lon = float(metadata[13])
    sq_km_area = int(metadata[14])
    orientation = int(metadata[15])
    return footprint, zoom, row, col, year, nadir_lat, nadir_lon, sq_km_area, orientation


with open("data/metadata_database_images.csv") as file:
    reader = csv.reader(file, delimiter=",")
    next(reader, None)  # skip the headers
    images_metadata = [row for row in reader]

with Pool(processes=NUM_PROCESSES) as pool:
    for _ in tqdm(pool.imap_unordered(download_region, images_metadata), total=len(images_metadata), desc="Downloading DB"):
        pass
