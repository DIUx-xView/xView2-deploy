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
from skimage.io import imread, imsave
from yacs.config import CfgNode

from tqdm import tqdm

from models.dual_hrnet import get_model
from xview2 import XView2Dataset
from utils import safe_mkdir

multiprocessing.set_start_method('spawn', True)
warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default="",
                    help='the location of the data folder')
parser.add_argument('--data_path', type=str, required=True, default="/mnt/Dataset/xView2/v2",
                    help='the location of the data folder')
parser.add_argument('--ckpt_path', type=str, required=True, default="",
                    help='Path to checkpoint')
parser.add_argument('--result_dir', type=str, required=True, default="",
                    help='Path to save result submit and compare iamges')
parser.add_argument('--is_train_data', action='store_true', dest='is_train_data',
                    help='')
parser.add_argument('--is_use_gpu', action='store_true', dest='is_use_gpu',
                    help='')

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


def main():
    if args.config_path:
        with open(args.config_path, 'rb') as fp:
            config = CfgNode.load_cfg(fp)
    else:
        config = None

    ckpt_path = args.ckpt_path
    result_submit_dir = os.path.join(args.result_dir, 'submit/')
    result_compare_dir = os.path.join(args.result_dir, 'compare/')
    dataset_mode = 'test' if not args.is_train_data else 'train'
    imgs_dir = os.path.join(args.data_path, 'test/images/') if dataset_mode == 'test' \
        else os.path.join(args.data_path, 'tier3/images/')

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    print('data folder: ', args.data_path)

    safe_mkdir(result_submit_dir)
    safe_mkdir(result_compare_dir)

    model = get_model(config)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['state_dict'])
    model.eval()
    model_wrapper = ModelWraper(model, args.is_use_gpu, config.MODEL.IS_SPLIT_LOSS)
    # model_wrapper = nn.DataParallel(model_wrapper)
    model_wrapper.eval()

    testset = XView2Dataset(args.data_path, rgb_bgr='rgb', preprocessing={'flip': False, 'scale': None, 'crop': None},
                            mode=dataset_mode)
    testset_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, pin_memory=False, num_workers=1)

    for i, samples in enumerate(tqdm(testset_loader)):
        if dataset_mode == 'train' and i < 5520:
            continue
        inputs_pre = samples['pre_img']
        inputs_post = samples['post_img']
        image_ids = samples['image_id']

        loc, cls = model_wrapper(inputs_pre, inputs_post)

        if config.MODEL.IS_SPLIT_LOSS:
            loc, cls = argmax(loc, cls)
            loc = loc.detach().cpu().numpy().astype(np.uint8)
            cls = cls.detach().cpu().numpy().astype(np.uint8)
        else:
            loc = torch.argmax(loc, dim=1, keepdim=False)
            loc = loc.detach().cpu().numpy().astype(np.uint8)
            cls = copy.deepcopy(loc)

        for image_id, l, c in zip(image_ids, loc, cls):
            localization_filename = 'test_localization_%s_prediction.png' % image_id
            damage_filename = 'test_damage_%s_prediction.png' % image_id

            imsave(os.path.join(result_submit_dir, localization_filename), l)
            imsave(os.path.join(result_submit_dir, damage_filename), c)

            pre_filename = 'test_pre_%s.png' % image_id
            post_filename = 'test_post_%s.png' % image_id
            pre_image = imread(os.path.join(imgs_dir, pre_filename))
            post_image = imread(os.path.join(imgs_dir, post_filename))

            mask_map_img = np.zeros((c.shape[0], c.shape[1], 3), dtype=np.uint8)
            mask_map_img[c == 1] = (255, 255, 255)
            mask_map_img[c == 2] = (229, 255, 50)
            mask_map_img[c == 3] = (255, 159, 0)
            mask_map_img[c == 4] = (255, 0, 0)
            compare_img = np.concatenate((pre_image, mask_map_img, post_image), axis=1)

            compare_filename = 'test_%s.png' % image_id
            imsave(os.path.join(result_compare_dir, compare_filename), compare_img)


if __name__ == '__main__':
    main()
