import os
from urllib.request import urlretrieve
from tarfile import TarFile
from random import shuffle
from glob import glob
from PIL import Image
import numpy as np
from tensorflow import keras


class Dataset(keras.utils.Sequence):
    """Dataset class implemented by Keras API

    Attributes:
        image_filenames: List containing all image file names
        annotation_filenames: List containing all annotation file names
            for each image
        num_classes: Number of classes.
        batch_size: Dataset batch size
    """

    def __init__(self, image_filenames, annotation_filenames, num_classes,
                 batch_size):
        self.num_classes = num_classes
        self.image_filenames = image_filenames
        self.annotation_filenames = annotation_filenames
        self.batch_size = batch_size
        self.dataset = self.get_file_path()

    def get_file_path(self):
        """This function gets the path to each image from the dataset.

        Returns:
            dataset: List containing images and annotations, divided into
                batches. List structure:
                    [
                        # First batch
                        [
                            [image_filename1, image_filename2, ...],
                            [annotation_filename1, annotation_filename2, ...]
                        ]

                        # Second batch
                        [
                            [image_filename1, image_filename2, ...],
                            [annotation_filename1, annotation_filename2, ...]
                        ]
                        ...
                    ]
        """
        dataset = []

        for i in range(len(self.image_filenames) // self.batch_size):
            dataset.append([
                self.image_filenames[i * self.batch_size:
                                     (i + 1) * self.batch_size],
                self.annotation_filenames[i * self.batch_size:
                                          (i + 1) * self.batch_size]
            ])
        if len(self.image_filenames) % self.batch_size != 0:
            idx = len(self.image_filenames) // self.batch_size
            dataset.append([self.image_filenames[idx * self.batch_size:],
                            self.annotation_filenames[idx * self.batch_size:]])
        return dataset

    def annotation_processing(self, annotations):
        """This function changes the mask of images, turning each
        pixel into one-hot vector.

        Args:
            annotations: NumPy array containing images. Array shape:
                (num_images, img_height, img_width)

        Returns:
            vectorized_annotation: NumPy array with shape
                (num_images, img_height, img_width, num_classes)
        """
        annotations = np.where(annotations == 255, 0, annotations)
        annotations = np.expand_dims(annotations, -1)
        vectorized_annotation = np.array(
            np.equal(annotations, np.arange(self.num_classes)),
            dtype="float32")
        return vectorized_annotation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        # Due to the internal structure of the neural network,
        # the image size must be a multiple of 32 and more than 256
        image_shape = np.array(Image.open(self.dataset[idx][0][0])).shape
        image_shape = [max(image_shape[0] // 32 * 32, 256),
                       max(image_shape[1] // 32 * 32, 256), image_shape[2]]
        images = np.zeros([0] + image_shape, "float32")

        # The neural network can be trained on images of different sizes,
        # so all images from the package must first resize to the same
        # size. To do this, all images and annotations change their size
        # and make it equal to the size of the first image of the package.
        for image_filename in self.dataset[idx][0]:
            image = Image.open(image_filename).resize([image_shape[1],
                                                       image_shape[0]])
            image = np.expand_dims(np.array(image, "float32"), 0)
            images = np.append(images, image, 0)
        annotations = np.zeros([0] + image_shape[:2], "float32")
        for annotation_filename in self.dataset[idx][1]:
            annotation = Image.open(annotation_filename).resize(
                [image_shape[1], image_shape[0]])
            annotation = np.array(annotation, "float32")
            annotation = np.expand_dims(annotation, 0)
            annotations = np.append(annotations, annotation, 0)
        annotations = self.annotation_processing(annotations)
        images /= 255
        return images, annotations


def get_filenames(folder):
    """This function get image and annotation file names from a dataset.

    Args:
        folder: Folder with PASCAL VOC 2012 dataset. This directory should
            contain a folder "VOCdevkit"

    Returns:
        images: List containing all images from dataset
        annotations: List containing segmented mask for each image
    """
    annotations = glob(f"{folder}VOCdevkit/VOC2012/SegmentationClass/*")
    shuffle(annotations)
    image_codes = [elem.split('/')[-1].split(".")[0] for elem in annotations]
    images = [f"{folder}VOCdevkit/VOC2012/JPEGImages/{image_code}.jpg"
              for image_code in image_codes]
    return images, annotations


def download_dataset(download_folder):
    """This function loads the PASCAL VOC dataset into the specified directory.

    Args:
        download_folder: Folder with dataset. After downloading, the directory
            "VOCdevkit" will be created in this folder.
    """
    urlretrieve(
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/"
        "VOCtrainval_11-May-2012.tar", "dataset.tar"
    )
    TarFile("dataset.tar").extractall(download_folder)
    os.remove("dataset.tar")
