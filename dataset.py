import os
from skimage.io import imread
import tifffile
from torch.utils.data import Dataset    
   
class XViewDataset(Dataset):
    "Dataset for xView"
    
    def __init__(self, pairs, config, transform=None, return_geo=False):
        """
        :param pre_chips: List of pre-damage chip filenames
        :param post_chips: List of post_damage chip filenames
        :param transform: PyTorch transforms to be used on each example
        """
        self.pairs = pairs
        self.transform=transform
        self.config=config
        self.return_geo=return_geo
        
        
    def __len__(self):
        return(len(self.pairs))
    
    def __getitem__(self, idx):
        fl = self.pairs[idx]
        if self.config.DATASET.IS_TIFF:
            pre_image = tifffile.imread(fl.opts.in_pre_path)
            post_image = tifffile.imread(fl.opts.in_post_path)
        else:
            pre_image = imread(fl.opts.in_pre_path)
            post_image = imread(fl.opts.in_post_path)
        
        out_dict = {}
        out_dict['pre_image']=self.transform(pre_image)
        out_dict['post_image']=self.transform(post_image)
        out_dict['idx']=idx
        out_dict['out_cls_path'] = str(fl.opts.out_cls_path)
        out_dict['out_loc_path'] = str(fl.opts.out_loc_path)
        out_dict['out_overlay_path'] = str(fl.opts.out_overlay_path) 
        out_dict['is_vis'] = fl.opts.is_vis
        
        return out_dict
    
class XViewDataset2(Dataset):
    "Dataset for xView"
    
    def __init__(self, pre_chips, post_chips, args, config, transform=None):
        """
        :param pre_chips: List of pre-damage chip filenames
        :param post_chips: List of post_damage chip filenames
        :param transform: PyTorch transforms to be used on each example
        """
        self.pre_directory=args.pre_directory,
        self.post_directory=args.post_directory,
        self.pre_chips=pre_chips
        self.post_chips=post_chips
        self.transform=transform
        self.config=config
        
        assert len(pre_chips) == len(post_chips)
        
    def __len__(self):
        return(len(self.pre_chips))
    
    def __getitem__(self, idx):
        pre=os.path.join(self.pre_directory, self.pre_chips[idx])
        post=os.path.join(self.post_directory,self.post_chips[idx])
        if self.config.DATASET.IS_TIFF:
            pre_image = tifffile.imread(args.in_pre_path)
            post_image = tifffile.imread(args.in_post_path)
        else:
            pre_image = imread(args.in_pre_path)
            post_image = imread(args.in_post_path)
        
        pre= self.transform(pre)
        post=self.transform(post)
        
        return pre , post