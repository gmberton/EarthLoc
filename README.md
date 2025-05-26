# EarthLoc
Code for CVPR 2024 paper "EarthLoc: Astronaut Photography Localization by Indexing Earth from Space".
The paper introduces the task of Astronaut Photography Localization (APL) through image retrieval.

[[ArXiv](https://arxiv.org/abs/2403.06758)] [[BibTex](https://github.com/gmberton/EarthLoc?tab=readme-ov-file#cite)] [[Webpage](https://earthloc-and-earthmatch.github.io/)]

<p  align="center">
  <img src="https://github.com/EarthLoc-and-EarthMatch/EarthLoc-and-EarthMatch.github.io/blob/b0902f64ef548ee1e3e5d5fdbda3c99e7ef27146/static/images/task_animation_low_res.gif" width="60%"/>
</p>


## Setup
Clone the repo and install packages
```
git clone https://github.com/gmberton/EarthLoc
cd EarthLoc
pip install -r requirements.txt
```

## Data
To download the data you can simply run
```
rsync -rhz --info=progress2 --ignore-existing rsync://vandaldata.polito.it/sf_xl/EarthLoc/data .
```
Note: the data can only be downloaded with rsync. Using wget or HTTP or trying to download from your browser is not going to work.

This will download all required images within the directory `data`, which will take about 65 GB in storage.

Each image filename contains its metadata, according to this format:
`
@ lat1 @ lon1 @ lat2 @ lon2 @ lat3 @ lon3 @ lat4 @ lon4 @ image_id @ timestamp @ nadir_lat @ nadir_lon @ sq_km_area @ orientation @ .jpg
`

Where the first 8 fields are the latitudes and longitudes of the 4 corners of the image (i.e. the footprint). `nadir_lat` and `nadir_lon` are the position of nadir (which corresponds to the center of the footprint in database images, but can be thousands of kilometers aways from the footprint for queries).

For database images, `image_id` corresponds to zoom, row, column (according to WMTS).
For query images, `image_id` corresponds to mission, roll, frame, which are a unique identifier of ISS photographs.

`sq_km_area` is the footprint covered area in squared kilometers, and `orientation` is the orientation of the image from 0 to 360° (e.g. 0° means that the image is north-up, like a normal map): orientation is always 0° for database images.

Besides the images, the rsync command will download the file containing the intersections between queries and database images (it is faster to have them pre-computed than to compute them online at the beginning of every run) in `data/queries_intersections_with_db_2021.pt`.

## Train
Once the dataset is downloaded, simply run
```
python train.py
```
or
```
python train.py -h
```
to see the possible hyperparameters.


## Trained model

Available [here](https://drive.google.com/file/d/1NJUVZm6-JncHRR01pjj4QjWNYjcLbIzm/view?usp=drive_link)


## Cite
Here is the bibtex to cite our paper
```
@InProceedings{Berton_CVPR_2024_EarthLoc,
    author    = {Berton, Gabriele and Stoken, Alex and Caputo, Barbara and Masone, Carlo},
    title     = {EarthLoc: Astronaut Photography Localization by Indexing Earth from Space},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
}
```
