from argparse import ArgumentParser

import glob
import os

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12

import sys
sys.path.append('/home/kirepreshanth/Documents/Dissertation/generative-inpainting-pytorch') # TODO allow this to be passed as part of the args.

from PIL import Image
import numpy as np

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from generator import generateInpaintedImage


# TODO Move these to a uitls file.
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
    plt.figure()
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.show()
    
def creatDirectoryIfNotExists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='/home/kirepreshanth/Documents/Dissertation/generative-inpainting-pytorch/checkpoints/cityscapes/hole_benchmark/config.yaml',
                    help="training configuration")
parser.add_argument('--seed', type=int, help='manual seed')
parser.add_argument('--checkpoint_path', type=str, default='/home/kirepreshanth/Documents/Dissertation/generative-inpainting-pytorch/checkpoints/cityscapes/hole_benchmark', help='Path to the checkpoints where the latest GAN model is saved.')
parser.add_argument('--folder', type=str, default='/home/kirepreshanth/Documents/Dissertation/datasets/cityscapes/occluded_dataset/leftImg8bit_co/val', help='Folder containing dataset with occlusions.')
parser.add_argument('--mask', type=str, default='/home/kirepreshanth/Documents/Dissertation/generative-inpainting-pytorch/examples/center_mask_256.png')
parser.add_argument('--output', type=str, default='/home/kirepreshanth/Documents/Dissertation/generative-inpainting-pytorch/examples/output_1.png')
parser.add_argument('--flow', type=str, default='')
parser.add_argument('--iter', type=int, default=0)

args = parser.parse_args()

for filePath in glob.glob(args.folder + "/**/*.png"):
    
    pathParts = filePath.split('/')
    
    fileName = pathParts[-1]
    city = pathParts[-2]
        
    name, ext = fileName.split('.')
    
    fileName_occluded_instanceSegmented = name + "_ois." + ext
    fileName_filled_IS = name + "_fis." + ext

    image_occluded = load(filePath)

    inpainted_image = generateInpaintedImage(args, filePath)
    inpainted_image = np.array(inpainted_image)[:, :, [2, 1, 0]]

    config_file = "/home/kirepreshanth/Documents/Dissertation/config.yaml"

    # This updates the config options with the same 
    # config file used to train the MaskRCNN
    cfg.merge_from_file(config_file)

    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
    )

    predictions_occluded = coco_demo.run_on_opencv_image(image_occluded)
    predictions_filled = coco_demo.run_on_opencv_image(inpainted_image)
    
    pathParts[-3] = pathParts[-3] + '_is_128'
    segmented_folder = '/'.join(pathParts[:-2])
    
    creatDirectoryIfNotExists(segmented_folder)
    city_folder = segmented_folder + '/' + city
    creatDirectoryIfNotExists(city_folder)
    
    predictions_occluded = Image.fromarray(predictions_occluded[:, :, [2, 1, 0]])
    predictions_filled = Image.fromarray(predictions_filled[:, :, [2, 1, 0]])
    
    predictions_occluded.save(city_folder + '/' + fileName_occluded_instanceSegmented)
    predictions_filled.save(city_folder + '/' + fileName_filled_IS)
    
    #imshow(predictions_occluded)
    #imshow(predictions_filled)