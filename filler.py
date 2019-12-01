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
import yaml

from generator_filler import generateInpaintedImage


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
        
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='/home/kirepreshanth/Documents/Dissertation/generative-inpainting-pytorch/checkpoints/cityscapes/hole_benchmark/config_lr1e5.yaml',
                    help="training configuration")
parser.add_argument('--seed', type=int, help='manual seed')
parser.add_argument('--checkpoint_path', type=str, default='/home/kirepreshanth/Documents/Dissertation/generative-inpainting-pytorch/checkpoints/cityscapes/hole_benchmark', help='Path to the checkpoints where the latest GAN model is saved.')
parser.add_argument('--folder', type=str, default='/home/kirepreshanth/Documents/Dissertation/datasets/cityscapes/occluded_dataset/leftImg8bit_to/val', help='Folder containing dataset with occlusions.')
parser.add_argument('--mask', type=str, default='/home/kirepreshanth/Documents/Dissertation/generative-inpainting-pytorch/examples/center_mask_256.png')
parser.add_argument('--flow', type=str, default='')
parser.add_argument('--iter', type=int, default=0)

args = parser.parse_args()

for filePath in glob.glob(args.folder + "/**/*.png"):
    
    pathParts = filePath.split('/')
    
    fileName = pathParts[-1]
    city = pathParts[-2]
        
    name, ext = fileName.split('.')
    
    image_occluded = load(filePath)

    inpainted_image = generateInpaintedImage(args, filePath)
    inpainted_image = np.array(inpainted_image)[:, :, [2, 1, 0]]
    
    config = get_config(args.config)
    experiment_name = config['expname']
    architecture = config['netG']['architecture']
    
    pathParts[-3] = pathParts[-3] + '_lr1e5_' + architecture
    segmented_folder = '/'.join(pathParts[:-2])
    
    creatDirectoryIfNotExists(segmented_folder)
    city_folder = segmented_folder + '/' + city
    creatDirectoryIfNotExists(city_folder)
        
    inpainted_image = Image.fromarray(inpainted_image[:, :, [2, 1, 0]])
    save_location = city_folder + '/' + fileName
    inpainted_image.save(save_location)
    print("Saved the inpainted result to {}".format(save_location))