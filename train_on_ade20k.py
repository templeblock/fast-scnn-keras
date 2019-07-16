import os
from glob import glob
from urllib.request import urlretrieve
from zipfile import ZipFile
import numpy as np


def download_dataset(folder=None):
    """This function loads the ADE20K dataset into the specified directory.
    
    Args:
        folder: Directory where the dataset will be saved
    """
    dataset_url = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'
    urlretrieve(dataset_url, "dataset.zip")
    ZipFile("dataset.zip").extractall(folder)
    os.remove("dataset.zip")


def get_one_hot_vector(idx, length):
    """This function creates one-hot NumPy vector
    
    Args:
        idx: Index of non-zero element
        length: Vector length
    """
    assert idx < length
    return np.array([int(i == idx) for i in range(length)])


def get_file_path(folder="ADEChallengeData2016"):
    """This function gets the path to each image from the dataset.
    
    Args:
        folder: Folder to the directory with the dataset.
    
    Returns:
        train_dataset: List with lists. Each list containing the path to the
            image and the path to the annotation.
            List structure:
                [
                    [image_filename1, annotation_filename1],
                    [image_filename2, annotation_filename2],
                    [image_filename3, annotation_filename3],
                    ...
                ]

        val_dataset: List with lists. Each list containing the path to the
            image and the path to the annotation
            List structure:
                [
                    [image_filename1, annotation_filename1],
                    [image_filename2, annotation_filename2],
                    [image_filename3, annotation_filename3],
                    ...
                ]
    """
    train_images = sorted(glob(f"{folder}/images/training/*"))
    train_annotations = sorted(glob(f"{folder}/annotations/training/*"))
    train_dataset = [[train_images[i], train_annotations[i]]
                     for i in range(len(train_images))]
    
    val_images = sorted(glob(f"{folder}/images/validation/*"))
    val_annotations = sorted(glob(f"{folder}/annotations/validation/*"))
    val_dataset = [[val_images[i], val_annotations[i]]
                     for i in range(len(val_images))]
    return train_dataset, val_dataset
