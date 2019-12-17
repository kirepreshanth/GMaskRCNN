
########################################################################
# This script will run the GMask R-CNN on the occluded dataset provided.
# The code for the COCODemo in the predictor file came from:
# Facebook AI Research Team (2019) [sourcecode] 
# https://github.com/facebookresearch/maskrcnn-benchmark
# And remains unchanged.
########################################################################

import glob
import os
import sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from PIL import Image
from argparse import ArgumentParser

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12

## General functions used for loading images etc.

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
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    
def creatDirectoryIfNotExists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

## Arguement Parser

parser = ArgumentParser()
parser.add_argument('--g_config', type=str, default='models/deepfill/cityscapes/hole_benchmark/config.yaml',
                    help='Path to trained DeepFill config file location.')
parser.add_argument('--m_config', type=str, default='models/mask_rcnn/cityscapes_poly/config.yml',
                    help='Path to trained Mask R-CNN config file location.')
parser.add_argument('--gip_path', type=str, default='../generative-inpainting-pytorch', help='Location of the Generative Inpaiting pytorch repository')
parser.add_argument('--seed', type=int, help='manual seed')
parser.add_argument('--checkpoint_path', type=str, default='models/deepfill/cityscapes/hole_benchmark', help='Path to the checkpoints where the latest GAN model is saved.')
parser.add_argument('--folder', type=str, default='datasets/occluded_cityscapes/val', help='Folder containing dataset with occlusions.')
parser.add_argument('--mask', type=str, default='center_mask_512.png')
parser.add_argument('--save_output', type=bool, default=False)
parser.add_argument('--flow', type=str, default='')
parser.add_argument('--iter', type=int, default=0)

args = parser.parse_args()
sys.path.append(args.gip_path)

from generator import generateInpaintedImage, loadGenerator

# This updates the config options with the same 
# config file used to train the MaskRCNN
cfg.merge_from_file(args.m_config)

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)

# Load trained Generator from DeepFill model.
netG = loadGenerator(args)

# Loop through each image in the Occluded Dataset
for filePath in glob.glob(args.folder + "/**/*.png"):
    
    pathParts = filePath.split('/')    
    fileName = pathParts[-1]
    city = pathParts[-2]        
    name, ext = fileName.split('.')
    
    fileName_occluded_instanceSegmented = name + "_ois." + ext
    fileName_filled_IS = name + "_fis." + ext

    image_occluded = load(filePath)

    # Hallucinate the occlusions in the loaded images
    inpainted_image = generateInpaintedImage(args, netG, filePath)
    inpainted_image = np.array(inpainted_image)[:, :, [2, 1, 0]]

    # Feed images to the Mask R-CNN and produce instance segmentation annotations.
    predictions_occluded = coco_demo.run_on_opencv_image(image_occluded)
    predictions_filled = coco_demo.run_on_opencv_image(inpainted_image)
        
    predictions_occluded = Image.fromarray(predictions_occluded[:, :, [2, 1, 0]])
    predictions_filled = Image.fromarray(predictions_filled[:, :, [2, 1, 0]])
    
    # If you would like to save the output of the instance segmentation then
    # Set the 'save_output' argument to True.
    if args.save_output:
        pathParts[-3] = pathParts[-3] + '_is_256'
        segmented_folder = '/'.join(pathParts[:-2])
        
        creatDirectoryIfNotExists(segmented_folder)
        city_folder = segmented_folder + '/' + city
        creatDirectoryIfNotExists(city_folder)
    
        predictions_occluded.save(city_folder + '/' + fileName_occluded_instanceSegmented)
        predictions_filled.save(city_folder + '/' + fileName_filled_IS)
    else:
        imshow(predictions_occluded)
        imshow(predictions_filled)