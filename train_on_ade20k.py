import os
from zipfile import ZipFile
from urllib.request import urlretrieve


def download_dataset(folder):
    """This function loads the ADE20K dataset into the specified directory.
    """
    dataset_url = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'
    urlretrieve(dataset_url, "dataset.zip")
    ZipFile("dataset.zip").extractall(folder)
    os.remove("dataset.zip")
