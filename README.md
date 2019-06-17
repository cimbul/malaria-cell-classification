# Malaria detection with cell images

Experimenting with ML models to classify pre-segmented red blood cell images as uninfected or
infected with _P. falciparum_. Main work is kept in the [`Malaria.ipynb`](Malaria.ipynb) Jupyter
notebook.

## Data

Source data originally comes from the publication:

> Rajaraman S, Antani SK, Poostchi M, Silamut K, Hossain MA, Maude, RJ, Jaeger S, Thoma GR. (2018)
> Pre-trained convolutional neural networks as feature extractors toward improved Malaria parasite
> detection in thin blood smear images. PeerJ6:e4568 https://doi.org/10.7717/peerj.4568

The data was made available by NIH [here][NIH source]. It was posted on Kaggle by user Arunava at
[iarunava/cell-images-for-detecting-malaria].

[NIH source]: https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria
[iarunava/cell-images-for-detecting-malaria]: https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria

## Setup

Work was developed with Python 3.7 and JupyterLab 0.35. Core Python dependencies are listed in
[`requirements.txt`](./requirements.txt), and the full venv contents are listed in
[`requirements-freeze.txt`](./requirements-freeze.txt).

Assuming you have already set up Python, Jupyter, and, if desired, a venv and corresponding Jupyter
kernel:

```bash
$ pip install -r requirements.txt
$ jupyter lab
```
