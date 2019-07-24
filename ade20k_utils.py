from glob import glob
from PIL import Image
import numpy as np
from tensorflow import keras


class ADE20KDataset(keras.utils.Sequence):
    """Dataset class implemented by Keras API

    Attributes:
        images_folder: Folder with images
        annotations_folder: Folder with annotations to the images.
            Annotations will be matched with images by the number
            indicated in the image filename
        num_classes: Number of classes.
        batch_size: Dataset batch size
    """

    def __init__(self, images_folder, annotations_folder, num_classes,
                 batch_size):
        self.num_classes = num_classes
        self.images_folder = images_folder
        self.annotations_folder = annotations_folder
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
        images = sorted(glob(f"{self.images_folder}/*"))
        annotations = sorted(glob(f"{self.annotations_folder}/*"))
        dataset = []
        for i in range(len(images) // self.batch_size):
            dataset.append([
                images[i * self.batch_size: (i + 1) * self.batch_size],
                annotations[i * self.batch_size: (i + 1) * self.batch_size]
            ])
        if len(images) % self.batch_size != 0:
            idx = len(images) // self.batch_size
            dataset.append([images[idx * self.batch_size:],
                            annotations[idx * self.batch_size:]])
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
