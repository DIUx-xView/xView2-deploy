import os
import argparse
import multiprocessing
import warnings
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
from skimage.io import imread, imsave
from yacs.config import CfgNode

from models.dual_hrnet import get_model

multiprocessing.set_start_method('spawn', True)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('in_pre_path', type=str, default='test_images/test_pre_00000.png')
parser.add_argument('in_post_path', type=str, default='test_images/test_post_00000.png')
parser.add_argument('out_loc_path', type=str, default='test_images/test_loc_00000.png')
parser.add_argument('out_cls_path', type=str, default='test_images/test_cls_00000.png')
parser.add_argument('--model_config_path', type=str, default='configs/model.yaml')
parser.add_argument('--model_weight_path', type=str, default='weights/weight.pth')
parser.add_argument('--is_use_gpu', action='store_true', dest='is_use_gpu')
parser.add_argument('--is_vis', action='store_true', dest='is_vis')

args = parser.parse_args()


class ModelWraper(nn.Module):
    def __init__(self, model, is_use_gpu=False, is_split_loss=True):
        super(ModelWraper, self).__init__()
        self.is_use_gpu = is_use_gpu
        self.is_split_loss = is_split_loss
        if self.is_use_gpu:
            self.model = model.cuda()
        else:
            self.model = model

    def forward(self, inputs_pre, inputs_post):
        inputs_pre = Variable(inputs_pre)
        inputs_post = Variable(inputs_post)

        if self.is_use_gpu:
            inputs_pre = inputs_pre.cuda()
            inputs_post = inputs_post.cuda()

        pred_dict = self.model(inputs_pre, inputs_post)
        loc = F.interpolate(pred_dict['loc'], size=inputs_pre.size()[2:4], mode='bilinear')

        if self.is_split_loss:
            cls = F.interpolate(pred_dict['cls'], size=inputs_post.size()[2:4], mode='bilinear')
        else:
            cls = None

        return loc, cls


def argmax(loc, cls):
    loc = torch.argmax(loc, dim=1, keepdim=False)
    cls = torch.argmax(cls, dim=1, keepdim=False)

    cls = cls + 1
    cls[loc == 0] = 0

    return loc, cls


def build_image_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def main():
    config = CfgNode.load_cfg(open(args.model_config_path, 'rb'))
    ckpt_path = args.model_weight_path

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    model = get_model(config)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['state_dict'])
    model.eval()

    model_wrapper = ModelWraper(model, args.is_use_gpu, config.MODEL.IS_SPLIT_LOSS)
    model_wrapper.eval()

    image_transforms = build_image_transforms()

    pre_image = imread(args.in_pre_path)
    post_image = imread(args.in_post_path)

    inputs_pre = image_transforms(pre_image)
    inputs_post = image_transforms(post_image)
    inputs_pre.unsqueeze_(0)
    inputs_post.unsqueeze_(0)

    loc, cls = model_wrapper(inputs_pre, inputs_post)

    if config.MODEL.IS_SPLIT_LOSS:
        loc, cls = argmax(loc, cls)
        loc = loc.detach().cpu().numpy().astype(np.uint8)[0]
        cls = cls.detach().cpu().numpy().astype(np.uint8)[0]
    else:
        loc = torch.argmax(loc, dim=1, keepdim=False)
        loc = loc.detach().cpu().numpy().astype(np.uint8)[0]
        cls = copy.deepcopy(loc)

    imsave(args.out_loc_path, loc)
    imsave(args.out_cls_path, cls)

    if args.is_vis:
        mask_map_img = np.zeros((cls.shape[0], cls.shape[1], 3), dtype=np.uint8)
        mask_map_img[cls == 1] = (255, 255, 255)
        mask_map_img[cls == 2] = (229, 255, 50)
        mask_map_img[cls == 3] = (255, 159, 0)
        mask_map_img[cls == 4] = (255, 0, 0)
        compare_img = np.concatenate((pre_image, mask_map_img, post_image), axis=1)

        out_dir = os.path.dirname(args.out_loc_path)
        imsave(os.path.join(out_dir, 'compare_img.png'), compare_img)


if __name__ == '__main__':
    main()
