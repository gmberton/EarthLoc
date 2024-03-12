
import csv
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool

from data_utils import download_image_from_url, image_save_atomically

NUM_PROCESSES = 4


def read_metadata(metadata):
    footprint = (
        float(metadata[0]), float(metadata[1]), float(metadata[2]), float(metadata[3]),
        float(metadata[4]), float(metadata[5]), float(metadata[6]), float(metadata[7]),
    )
    mission = metadata[8].strip()
    roll = metadata[9].strip()
    frame = metadata[10].strip()
    date = int(metadata[11])
    nadir_lat = float(metadata[12])
    nadir_lon = float(metadata[13])
    sq_km_area = int(metadata[14])
    orientation = metadata[15]
    url = metadata[16]
    return footprint, mission, roll, frame, date, nadir_lat, nadir_lon, sq_km_area, orientation, url


def download_image(metadata):
    footprint, mission, roll, frame, date, nadir_lat, nadir_lon, sq_km_area, orientation, url = read_metadata(metadata)
    image_name = \
        f"@{footprint[0]:.6f}@{footprint[1]:.6f}@{footprint[2]:.6f}@{footprint[3]:.6f}" \
        f"@{footprint[4]:.6f}@{footprint[5]:.6f}@{footprint[6]:.6f}@{footprint[7]:.6f}" \
        f"@{mission}-{roll}-{frame}@{date}@{nadir_lat:.1f}@{nadir_lon:.1f}" \
        f"@{sq_km_area}@{orientation}@.jpg"
    image_path = folder_name / image_name
    if image_path.exists():
        return
    pil_img = download_image_from_url(url)
    pil_img = pil_img.resize([1024, 1024])
    image_save_atomically(pil_img, image_path)


with open("data/metadata_queries_images.csv") as file:
    reader = csv.reader(file, delimiter=",")
    next(reader, None)  # skip the headers
    images_metadata = [row for row in reader]

folder_name = Path("data") / "queries"

with Pool(processes=NUM_PROCESSES) as pool:
    for _ in tqdm(pool.imap_unordered(download_image, images_metadata), total=len(images_metadata), desc="Downloading queries"):
        pass

