# GMaskRCNN

This is one of the outputs produced from my Individual Project module

## Dependencies
This project is dependent on the following Repositories:
 - [maskrcnn-benchmark](https://github.com/kirepreshanth/maskrcnn-benchmark) (forked from [facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark))
 - [generative-inpainting-pytorch](https://github.com/kirepreshanth/generative-inpainting-pytorch) (forked from [DAA233/generative-inpainting-pytorch](https://github.com/DAA233/generative-inpainting-pytorch))

## Instructions
In order to run the GMask R-CNN you will need to first clone the repositories in the **Dependecies section** locally and follow these [instructions](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md) to install the Mask R-CNN benchmark only to a new conda environment

**Important: Installation is required to load the Mask R-CNN Model.**

Once installed, please download the [occluded Cityscapes dataset](https://drive.google.com/open?id=1hzZWZ6uKKuPX8B6Kjg6hcyXoxm_a3o-H)(1GB) and the [trained models](https://drive.google.com/open?id=1UVtDLYlLfHjKHCFJJB12v70lTA_gjecW) (7GB).
These are large files and make take some time to download.

Extract the models to the root of the GMaskRCNN local repository and the occluded dataset into the **datasets** folder also found in the root.

After extraction, open the local [run.py](https://github.com/kirepreshanth/GMaskRCNN/blob/master/run.py) file in a text editor or IDE, ensure that the default arguments are set to the correct values. The default values should work as they are. 
The only default that may need to be changed is the `--gip_path` argument, this should point to the location of your local `generative-inpainting-pytorch` repository (root).

Finally you can run the run.py file this will run through each image in the occluded dataset and run instance segmentation on them, the first window shown will be the original image the second will be the image with the occlusions hallucinated.
