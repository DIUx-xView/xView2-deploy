import os
import argparse
import multiprocessing
import warnings
import copy
from collections import defaultdict

import rasterio
import numpy as np
import tifffile
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
from skimage.io import imread, imsave
from yacs.config import CfgNode

from models.dual_hrnet import get_model
from utils import build_image_transforms

multiprocessing.set_start_method('spawn', True)
warnings.filterwarnings("ignore")


class Options(object):

    def __init__(self, pre_path='input/pre', post_path='input/post',
                 out_loc_path='output/loc', out_dmg_path='output/dmg', out_overlay_path='output/over',
                 model_config='configs/model.yaml', model_weights='weights/weight.pth',
                 geo_profile=None, use_gpu=False, vis=False):
        self.in_pre_path = pre_path
        self.in_post_path = post_path
        self.out_loc_path = out_loc_path
        self.out_cls_path = out_dmg_path
        self.out_overlay_path = out_overlay_path
        self.model_config_path = model_config
        self.model_weight_path = model_weights
        self.geo_profile = geo_profile
        self.is_use_gpu = use_gpu
        self.is_vis = vis

def parse_cli_args():
    # TODO: Unsure if this works. Moved to a function to allow direct passing of args.
    parser = argparse.ArgumentParser()
    parser.add_argument('in_pre_path', type=str, default='test_images/test_pre_00000.png')
    parser.add_argument('in_post_path', type=str, default='test_images/test_post_00000.png')
    parser.add_argument('out_loc_path', type=str, default='test_images/test_loc_00000.png')
    parser.add_argument('out_cls_path', type=str, default='test_images/test_cls_00000.png')
    parser.add_argument('--model_config_path', type=str, default='configs/model.yaml')
    parser.add_argument('--model_weight_path', type=str, default='weights/weight.pth')
    parser.add_argument('--is_use_gpu', action='store_true', dest='is_use_gpu')
    parser.add_argument('--is_vis', action='store_true', dest='is_vis')

    return parser.parse_args()


class ModelWrapper(nn.Module):
    def __init__(self, model, is_use_gpu=False, is_split_loss=True):
        super(ModelWrapper, self).__init__()
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
    dm = len(loc.shape)-3 # handles cases where batch size is passed and is not
    loc = torch.argmax(loc, dim=dm, keepdim=False)
    cls = torch.argmax(cls, dim=dm, keepdim=False)

    cls = cls + 1
    cls[loc == 0] = 0

    return loc, cls


def run_inference(args, config, model_wrapper, eval_dataset, eval_dataloader):
    results = defaultdict(list)
    with torch.no_grad(): # This is really important to not explode memory with gradients!
        for result_dict in tqdm(eval_dataloader, total=len(eval_dataloader)):
            loc, cls = model_wrapper(result_dict['pre_image'], result_dict['post_image'])
            loc = loc.detach().cpu()
            cls = cls.detach().cpu()

            result_dict['pre_image'] = result_dict['pre_image'].cpu().numpy()
            result_dict['post_image'] = result_dict['post_image'].cpu().numpy()
            result_dict['loc'] = loc
            result_dict['cls'] = cls
            # Do this one separately because you can't return a class from a dataloader
            result_dict['geo_profile'] = [eval_dataset.pairs[idx].opts.geo_profile
                                          for idx in result_dict['idx']]
            for k,v in result_dict.items():
                results[k] = results[k] + list(v)

    # Making a list
    results_list = [dict(zip(results,t)) for t in zip(*results.values())]

    return results_list
