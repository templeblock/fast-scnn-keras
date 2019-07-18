from PIL import Image
from glob import glob
import numpy as np
from tensorflow.keras.utils import Sequence


class ADE20KDataset(Sequence):
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
        self.length = len(self.dataset)
    
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
            dataset.append([images[(i + 1) * self.batch_size:],
                            annotations[(i + 1) * self.batch_size:]])
        return dataset

    def get_one_hot_vector(self, idx, length):
        """This function creates one-hot NumPy vector
        
        Args:
            idx: Index of non-zero element
            length: Vector length
        """
        assert idx < length
        return np.array([int(i == idx) for i in range(length)])
    
    def annotation_processing(self, annotations):
        """This function changes the mask of images, turning each 
        pixel into one-hot vector.
        
        Args:
            annotations: NumPy array containing images. Array shape: 
                (num_images, img_height, img_width)
            num_classes: number of classes
                
        Returns:
            vectorized_annotation: NumPy array with shape 
                (num_images, img_height, img_width, num_classes)
        """
        vectorized_annotation = np.zeros((annotations.shape[0], 
                                          annotations.shape[1], 
                                          annotations.shape[2],
                                          self.num_classes))
        for i in range(annotations.shape[0]):
            for j in range(annotations.shape[1]):
                for k in range(annotations.shape[2]):
                    vectorized_annotation[i, j, k] = self.get_one_hot_vector(
                        int(annotations[i, j, k]), self.num_classes)
        return vectorized_annotation
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        image_size = None
        images = None
        for image_filename in self.dataset[idx][0]:
            if image_size:
                image = Image.open(image_filename).resize(
                    [image_size[1], image_size[0]])
                image = np.array(image, "float32")
                image = np.expand_dims(image, 0)
                images = np.append(images, image, 0)
            else:
                image = np.array(Image.open(image_filename), "float32")
                image_size = image.shape[:2]
                images = np.zeros([0] + list(image.shape), "float32")
        annotations = np.zeros([0] + list(images.shape[1:3]), "float32")
        for annotation_filename in self.dataset[idx][1]:
            annotation = Image.open(annotation_filename).resize(
                [image_size[1], image_size[0]])
            annotation = np.array(annotation, "float32")
            annotation = np.expand_dims(annotation, 0)
            annotations = np.append(annotations, annotation, 0)
        annotations = self.annotation_processing(annotations)
        return images, annotations
