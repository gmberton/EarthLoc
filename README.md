# EarthLoc
Code for CVPR 2024 paper "EarthLoc: Astronaut Photography Localization by Indexing Earth from Space".
The paper introduces the task of Astronaut Photography Localization (APL) through image retrieval.

[[ArXiv](https://arxiv.org/abs/2403.06758)] [[BibTex](https://github.com/gmberton/EarthLoc?tab=readme-ov-file#cite)]

## Setup
Clone the repo, install packages, and download the queries and database images as such
```
git clone https://github.com/gmberton/EarthLoc
cd EarthLoc
pip install -r requirements.txt

python data/database_downloader.py
python data/queries_downloader.py
```
This will download all required images within the directory `data`.

Then download the file containing the intersections between queries and database images (would be too heavy to compute online) [at this link](https://drive.google.com/file/d/169X9TnrWpdFy4WQpyBZ9DBFUwOb7Nl5F/view?usp=drive_link) and put it in `data/queries_intersections_with_db_2021.torch`.

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


## TODO / coming soon

- Automatize entire setup (download of data and `queries_intersections_with_db_2021.torch` if missing)
- Improve instructions to resume from trained model


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
