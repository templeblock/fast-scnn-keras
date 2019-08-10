from random import random
from PIL import Image
import argparse
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from model.model import create_fast_scnn

# Parse arguments from command line

parser = argparse.ArgumentParser(
    description="Creating segmentation maps using Fast-SCNN")
parser.add_argument("image", type=str, help="Path to image")
parser.add_argument("weights", type=str, help="Path to weights for Fast-SCNN")
parser.add_argument("--width", type=int, help="Image width", default=256)
parser.add_argument("--height", type=int, help="Image height", default=256)
args = parser.parse_args()

# Creating a neural network model

model = create_fast_scnn(num_classes=21,
                         input_shape=[args.height, args.width, 3])
model.load_weights(args.weights)

# Creating a prediction

img_size = [args.height, args.width]
img = Image.open(args.image)
img = img.resize([img_size[1], img_size[0]])
img = np.array(img)
colors = [[random() for i in range(3)] for j in range(21)]
classes = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
img = np.expand_dims(img / 255, 0)
unprocessed_prediction = np.argmax(model.predict(img)[0], -1)
prediction = np.zeros(list(unprocessed_prediction.shape) + [3])
for i in range(3):
    prediction[..., i] = np.vectorize(lambda x: colors[x][i])(
        unprocessed_prediction)

# Displaying result using MatPlotLib

plt.figure()
plt.axis("off")
plt.imshow(prediction)
legend_handles = []
for i in range(len(colors)):
    legend_handles.append(Patch(color=colors[i], label=classes[i]))
plt.legend(handles=legend_handles, loc=2)
plt.show()
