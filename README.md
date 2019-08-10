# Fast-SCNN
Implementing the Fast-SCNN neural network on Keras (Tensorflow 2.0 backend)<br>
Link to the oiginal paper: https://arxiv.org/pdf/1902.04502.pdf

### Table of contents:
 1. <a href='#required-libraries'>Required Libraries</a>
 2. <a href='#installation'>Installation</a>
 3. <a href='#training'>Training</a>
 4. <a href='#using'>Using</a>

### Required Libraries
 - Pillow
 - MatPlotLib
 - TensorFlow 2.0
 - NumPy

### Installation
```
git clone https://github.com/float256/fast-scnn-keras.git
unzip fast-scnn-keras-master.zip
```

### Training
To start training the neural network, write information about training to "config.json" file and run "train_on_pascal_voc.py". <br>Below is a brief description of the fields in "config.json":
   * **num_epochs:** Number of epochs,
   * **image_size:** A list containing the size of the image. The first element is height. The second element is the width,
   * **num_classes:** Number of classes,
   * **batch_size:** Batch size,
   * **val_size:** The size of the validation dataset. Number from 0 to 1,
   * **checkpoint_folder:** The folder in which all control points will be saved. Leave the field blank if checkpoints are not needed,
   * **folder_with_dataset:** The folder with the dataset. If there is no dataset in the folder, it will be automatically downloaded,
   * **learning_rate:** Learning rate. Number from 0 to 1
  
### Using
To use a neural network, run the project through the console and specify the path to the image and the path to the weights. You can also specify the width and height of the image (by default, both parameters are 256)

P.S. Neural network weights will be laid out later
