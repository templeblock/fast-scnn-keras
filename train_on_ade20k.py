import os
import json
from glob import glob
from zipfile import ZipFile
from urllib.request import urlretrieve
from tensorflow import keras
from ade20k_utils import ADE20KDataset
from model.model import create_fast_scnn, loss_function, custom_accuracy_metric


def download_dataset(download_folder):
    """This function loads the ADE20K dataset into the specified directory.

    Args:
        download_folder: Folder with dataset. After downloading, the directory
            “ADEChallengeData2016” will be created in this folder.
    """
    dataset_url = 'http://data.csail.mit.edu/places/ADEchallenge' \
                  '/ADEChallengeData2016.zip '
    urlretrieve(dataset_url, "dataset.zip")
    ZipFile("dataset.zip").extractall(download_folder)
    os.remove("dataset.zip")


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    # Loading and processing a dataset

    folder = config["folder_with_dataset"]
    if not (glob(f"{folder}ADEChallengeData2016")):
        download_dataset(folder)
    train_dataset = ADE20KDataset(
        images_folder=f"{folder}ADEChallengeData2016/images/training",
        annotations_folder=f"{folder}ADEChallengeData2016/annotations/training",
        num_classes=config["num_classes"],
        batch_size=config["batch_size"]
    )
    val_dataset = ADE20KDataset(
        images_folder=f"{folder}ADEChallengeData2016/images/validation",
        annotations_folder=f"{folder}ADEChallengeData2016/annotations/"
        "validation",
        num_classes=config["num_classes"],
        batch_size=config["batch_size"]
    )

    # Model training

    model = create_fast_scnn(num_classes=config["num_classes"],
                             input_shape=[None, None, 3])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss=loss_function,
                  metrics=[custom_accuracy_metric])
    if config["checkpoint_folder"]:
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                folder + "best_weights--{val_loss:.2f}.h5",
                verbose=1, save_best_only=True),
            keras.callbacks.ModelCheckpoint(
                folder + "epoch{epoch:03d}--{val_loss:.2f}.h5",
                verbose=1, period=1)
        ]
    else:
        callbacks = []
    model.fit_generator(generator=train_dataset, epochs=config["num_epochs"],
                        validation_data=val_dataset, callbacks=callbacks)
