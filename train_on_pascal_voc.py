import json
from glob import glob
from tensorflow import keras
from pascal_voc_utils import Dataset, get_filenames, download_dataset
from model.model import create_fast_scnn

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    # Loading and processing a dataset

    folder = config["folder_with_dataset"]
    if not (glob(f"{folder}VOCdevkit")):
        download_dataset(folder)
    images, annotations = get_filenames(folder)
    split_point = int(len(images) * config["val_size"])
    train_dataset = Dataset(
        image_filenames=images[split_point:],
        annotation_filenames=annotations[split_point:],
        num_classes=config["num_classes"],
        batch_size=config["batch_size"]
    )
    val_dataset = Dataset(
        image_filenames=images[:split_point],
        annotation_filenames=annotations[:split_point],
        num_classes=config["num_classes"],
        batch_size=config["batch_size"]
    )

    # Model training

    model = create_fast_scnn(num_classes=config["num_classes"],
                             input_shape=[None, None, 3])
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy")
    if config["checkpoint_folder"]:
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                config[
                    "checkpoint_folder"] + "best_weights--{val_loss:.2f}.hdf5",
                verbose=1, save_best_only=True, save_weights_only=True),
            keras.callbacks.ModelCheckpoint(
                config["checkpoint_folder"] + "epoch{epoch:03d}--{"
                                              "val_loss:.2f}.hdf5",
                verbose=1, period=1, save_weights_only=True),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
        ]
    else:
        callbacks = []
    model.fit_generator(generator=train_dataset, epochs=config["num_epochs"],
                        validation_data=val_dataset, callbacks=callbacks)
