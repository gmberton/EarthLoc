
import io
import time
import shutil
import random
from PIL import Image
from urllib.request import Request, urlopen


def download_image_from_url(url, num_tries=4):
    """Return RGB PIL Image from its URL"""
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


def image_save_atomically(pil_img, dst_img_path):
    """Save image in an atomic procedure, so that interrupting the process does
    not leave a corrupt image saved, so that image download can be interrupted
    safely at any moment."""
    dst_img_path.parent.mkdir(parents=True, exist_ok=True)
    # A hidden file to temporarily store the image during its creation
    tmp_img_path = "." + str(random.randint(0, 99999999999)) + dst_img_path.suffix
    pil_img.save(tmp_img_path)
    shutil.move(tmp_img_path, dst_img_path)
