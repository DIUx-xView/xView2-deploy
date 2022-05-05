from skimage.io import imread
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

def preprocess_inputs(x):
    x = np.asarray(x, dtype='float32')
    x /= 127
    x -= 1
    return x

class XViewDataset(Dataset):
    "Dataset for xView"

    def __init__(self, pairs, mode, return_geo=False):
        """
        :param pre_chips: List of pre-damage chip filenames
        :param post_chips: List of post_damage chip filenames
        :param transform: PyTorch transforms to be used on each example
        """
        self.pairs = pairs
        self.return_geo=return_geo
        self.mode = mode


    def __len__(self):
        return(len(self.pairs))

    def __getitem__(self, idx, return_img=False):
        fl = self.pairs[idx]
        pre_image = np.array(Image.open(str(fl.opts.in_pre_path)).convert('RGB'))
        post_image = np.array(Image.open(str(fl.opts.in_post_path)).convert('RGB'))
        if self.mode == 'cls':
            img = np.concatenate([pre_image, post_image], axis=2)
        elif self.mode == 'loc':
            img = pre_image
        else:
            raise ValueError('Incorrect mode!  Must be cls or loc')
            
        img = preprocess_inputs(img)

        inp = []
        inp.append(img)
        inp.append(img[::-1, ...])
        inp.append(img[:, ::-1, ...])
        inp.append(img[::-1, ::-1, ...])
        inp = np.asarray(inp, dtype='float')
        inp = torch.from_numpy(inp.transpose((0, 3, 1, 2))).float()
        
        out_dict = {}
        out_dict['in_pre_path'] = str(fl.opts.in_pre_path)
        out_dict['in_post_path'] = str(fl.opts.in_post_path)
        if return_img:
            out_dict['pre_image'] = pre_image
            out_dict['post_image'] = post_image
        out_dict['img'] = inp
        out_dict['idx'] = idx
        out_dict['out_cls_path'] = str(fl.opts.out_cls_path)
        out_dict['out_loc_path'] = str(fl.opts.out_loc_path)
        out_dict['out_overlay_path'] = str(fl.opts.out_overlay_path)
        out_dict['is_vis'] = fl.opts.is_vis

        return out_dict
