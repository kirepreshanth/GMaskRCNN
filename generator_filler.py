import os
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils

from model.networks import Generator
from utils.tools import get_config, random_bbox, mask_image, is_image_file, default_loader, normalize, get_model_list
from PIL import Image


def generateInpaintedImage(args, imagePath):
    config = get_config(args.config)
    occlusions = []

    # CUDA configuration
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = True

    print("Arguments: {}".format(args))

    # Set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random seed: {}".format(args.seed))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)

    print("Configuration: {}".format(config))

    try:  # for unexpected error logging
        with torch.no_grad():   # enter no grad context
            if is_image_file(imagePath):
                if args.mask and is_image_file(args.mask):
                    # Test a multiple masked image with a given mask
                    x = default_loader(imagePath)
                    x = transforms.Resize([512, 1024])(x)
                    
                    mask = default_loader(args.mask)
                    mask = transforms.Resize(config['image_shape'][:-1])(mask)
                    mask = transforms.CenterCrop(config['image_shape'][:-1])(mask)
                    mask = transforms.ToTensor()(mask)[0].unsqueeze(dim=0)
                    mask = mask.unsqueeze(dim=0)
                    
                    w, h = x.size
                    first = x.crop((0, 0, w//3, h))
                    second = x.crop((w//3, 0, ((w//3) * 2) + 2, h))
                    third = x.crop(((w//3) * 2, 0, w, h))
                    
                    for y in [first, second, third]:                    
                        y = transforms.CenterCrop(config['image_shape'][:-1])(y)
                        y = transforms.ToTensor()(y)
                        y = normalize(y)
                        y = y * (1. - mask)
                        #y = y.unsqueeze(dim=0)                        
                        occlusions.append(y)
                    
                elif args.mask:
                    raise TypeError("{} is not an image file.".format(args.mask))

                # Set checkpoint path
                if not args.checkpoint_path:
                    checkpoint_path = os.path.join('checkpoints',
                                                   config['dataset_name'],
                                                   config['mask_type'] + '_' + config['expname'])
                else:
                    checkpoint_path = args.checkpoint_path

                # Define the trainer
                netG = Generator(config['netG'], cuda, device_ids).cuda()
                # Resume weight
                last_model_name = get_model_list(checkpoint_path, "gen", iteration=args.iter)
                netG.load_state_dict(torch.load(last_model_name))
                model_iteration = int(last_model_name[-11:-3])
                print("Resume from {} at iteration {}".format(checkpoint_path, model_iteration))
                
                default_image = default_loader(imagePath)
                di_w, di_h = default_image.size
                
                for idx, occlusion in enumerate(occlusions):
                    if cuda:
                        netG = nn.parallel.DataParallel(netG, device_ids=device_ids)
                        occlusion = occlusion.cuda()
                        mask = mask.cuda()

                    # Inference
                    x1, x2, offset_flow = netG(occlusion, mask)
                    inpainted_result = x2 * mask + occlusion * (1. - mask)              

                    inp_hw = config['image_shape'][1]
                    
                    if idx == 0:
                        offset = ((di_w // 3 - inp_hw) // 2, (di_h - inp_hw) // 2)
                    elif idx == 1:
                        offset = ((di_w - inp_hw) // 2, (di_h - inp_hw) // 2)
                    elif idx == 2:
                        offset = ((((di_w - inp_hw) // 2) + (di_w // 3)), (di_h - inp_hw) // 2)          
                    
                    grid = vutils.make_grid(inpainted_result, normalize=True)                    
                        
                    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
                    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                    im = Image.fromarray(ndarr)
                    
                    im = transforms.CenterCrop(config['mask_shape'])(im)
                    im = transforms.Resize(config['image_shape'][:-1])(im)
                    default_image.paste(im, offset)
                                
                return default_image
            else:
                raise TypeError("{} is not an image file.".format)
        # exit no grad context
    except Exception as e:  # for unexpected error logging
        print("Error: {}".format(e))
        raise e