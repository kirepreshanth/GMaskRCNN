import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests # Not required
from io import BytesIO # Not required

# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12


from PIL import Image
import numpy as np

from maskrcnn_benchmark.config import cfg
from GMaskRCNN import GMaskRCNN
from predictor import COCODemo

config_file = "/home/kirepreshanth/Documents/Dissertation/config.yaml"

# This updates the config options with the same 
# config file used to train the MaskRCNN
cfg.merge_from_file(config_file)

# model = GMaskRCNN(cfg)
# test = model.getTest()
# print(test)

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)


def load(fileName):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    pil_image = Image.open(fileName).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.show()
    

image = load("test_image_2_full.png") #TODO need to test an image from the test set. NOT Training set
#imshow(image)

predictions = coco_demo.run_on_opencv_image(image)
imshow(predictions)