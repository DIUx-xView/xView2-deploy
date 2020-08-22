import glob
import argparse
from collections import defaultdict
from pathlib import Path

from functools import partial
import inference
import multiprocessing as mp
import numpy as np
from raster_processing import *
import rasterio.warp
import torch
from torch.utils.data import DataLoader
from yacs.config import CfgNode

from tqdm import tqdm

from dataset import XViewDataset
from models.dual_hrnet import get_model
from inference import ModelWrapper
from utils import build_image_transforms


# TODO: Clean up directory structure
# TODO: gather input and output files from folders --> create pre and post mosaic --> create intersection --> get chips from intersection for pre/post --> extract geotransform per chip --> hand off to inference --> georef outputs


class Files(object):

    def __init__(self, ident, pre_directory, post_directory, output_directory, pre, post):
        self.ident = ident
        self.pre = pre_directory.joinpath(pre).resolve()
        self.post = post_directory.joinpath(post).resolve()
        self.loc = output_directory.joinpath('loc').joinpath(f'{self.ident}.tif').resolve()
        self.dmg = output_directory.joinpath('dmg').joinpath(f'{self.ident}.tif').resolve()
        self.over = output_directory.joinpath('over').joinpath(f'{self.ident}.tif').resolve()
        self.profile = self.get_profile()
        self.transform = self.profile["transform"]
        self.opts = inference.Options(pre_path=self.pre,
                                      post_path=self.post,
                                      out_loc_path=self.loc,
                                      out_dmg_path=self.dmg,
                                      out_overlay_path=self.over,
                                      geo_profile=self.profile,
                                      vis=True,
                                      use_gpu=True
                                      )

    def get_profile(self):
        with rasterio.open(self.pre) as src:
            return src.profile

def make_staging_structure(staging_path):
    """
    Creates directory structure for staging.
    :param staging_path: Staging path
    :return: True if successful
    """
    Path(f"{staging_path}/pre").mkdir(parents=True, exist_ok=True)
    Path(f"{staging_path}/post").mkdir(parents=True, exist_ok=True)
    Path(f"{staging_path}/mosaics").mkdir(parents=True, exist_ok=True)

    return True


def make_output_structure(output_path):

    """
    Creates directory structure for outputs.
    :param output_path: Output path
    :return: True if succussful
    """

    Path(f"{output_path}/chips/pre").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/chips/post").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/loc").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/dmg").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/over").mkdir(parents=True, exist_ok=True)

    return True


def get_files(dirname, extensions=['.png', '.tif', '.jpg']):

    """
    Gathers list of files for processing from path recursively.
    :param dirname: path to parse
    :param extensions: extensions to match
    :return: list of files matching extensions
    """
    dir_path = Path(dirname)
    files = dir_path.glob('**/*')
    files = [path.resolve() for path in files]

    match = [f for f in files if f.suffix in extensions]
    return match


def reproject_helper(args, raster_tuple, procnum, return_dict):
    (pre_post, src_crs, raster_file) = raster_tuple
    basename = raster_file.stem
    dest_file = args.staging_directory.joinpath('pre').joinpath(f'{basename}.tif')
    try:
        return_dict[procnum] = (pre_post, reproject(raster_file, dest_file, src_crs, args.destination_crs))
    except ValueError:
        return None

    
def postprocess_and_write(result_dict, config):

        if config.MODEL.IS_SPLIT_LOSS:
            loc, cls = argmax(loc, cls)
            loc = loc.detach().cpu().numpy().astype(np.uint8)[0]
            cls = cls.detach().cpu().numpy().astype(np.uint8)[0]
        else:
            loc = torch.argmax(loc, dim=1, keepdim=False)
            loc = loc.detach().cpu().numpy().astype(np.uint8)[0]
            cls = copy.deepcopy(loc)

        geo_profile.update(dtype=rasterio.uint8)

        with rasterio.open(result_dict['out_loc_path'], 'w', **result_dict['geo_profile']) as dst:
            dst.write(loc, 1)

        with rasterio.open(result_dict['out_cls_path'], 'w', **result_dict['geo_profile']) as dst:
            dst.write(cls, 1)

        #imsave(args.out_loc_path, loc)
        #imsave(args.out_cls_path, cls)

        if result_dict['is_vis']:
            mask_map_img = np.zeros((cls.shape[0], cls.shape[1], 3), dtype=np.uint8)
            mask_map_img[cls == 1] = (255, 255, 255)
            mask_map_img[cls == 2] = (229, 255, 50)
            mask_map_img[cls == 3] = (255, 159, 0)
            mask_map_img[cls == 4] = (255, 0, 0)
            compare_img = np.concatenate((result_dict['pre_image'], mask_map_img, result_dict['post_image']), axis=1)

            out_dir = os.path.dirname(result_dict['out_overlay_path'])
            with rasterio.open(result_dict['out_overlay_path'], 'w', **result_dict['geo_profile']) as dst:
                # Go from (x, y, bands) to (bands, x, y)
                mask_map_img = np.flipud(mask_map_img)
                mask_map_img = np.rot90(mask_map_img, 3)
                mask_map_img = np.moveaxis(mask_map_img, [0, 1, 2], [2, 1, 0])
                dst.write(mask_map_img)

def main():
    parser = argparse.ArgumentParser(description='Create arguments for xView 2 handler.')

    parser.add_argument('--pre_directory', metavar='/path/to/pre/files/', type=Path)
    parser.add_argument('--post_directory', metavar='/path/to/post/files/', type=Path)
    parser.add_argument('--staging_directory', metavar='/path/to/staging/', type=Path)
    parser.add_argument('--output_directory', metavar='/path/to/output/', type=Path)
    parser.add_argument('--model_weight_path', metavar='/path/to/model/weights', type=Path)
    parser.add_argument('--model_config_path', metavar='/path/to/model/config', type=Path)
    parser.add_argument('--is_use_gpu', action='store_true', help="If True, use GPUs")
    parser.add_argument('--n_procs', default=4, help="Number of processors for multiprocessing", type=int)
    parser.add_argument('--pre_crs', help='The Coordinate Reference System (CRS) for the pre-disaster imagery.')
    parser.add_argument('--post_crs', help='The Coordinate Reference System (CRS) for the post-disaster imagery.')
    parser.add_argument('--destination_crs', metavar='EPSG:4326', help='The Coordinate Reference System (CRS) for the output overlays.')

    args = parser.parse_args()

    make_staging_structure(args.staging_directory)
    make_output_structure(args.output_directory)

    """
    pre_files = get_files(args.pre_directory)
    post_files = get_files(args.post_directory)
    print("Got files")
    
    print('Re-projecting...')

    # Run reprojection in parallel processes
    manager = mp.Manager()
    return_dict = manager.dict()
    jobs = []

    # Some data hacking to make it more efficient for multiprocessing
    pre_files = [("pre", args.pre_crs, x) for x in pre_files]
    post_files = [("post", args.post_crs, x) for x in post_files]
    files = pre_files + post_files

    # Launch multiprocessing jobs
    for idx, f in enumerate(files):
        p = mp.Process(target=reproject_helper, args=(args, f, idx, return_dict))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()

    reproj = [x for x in return_dict.values() if x[1] is not None]
    pre_reproj = [x[1] for x in reproj if x[0] == "pre"]
    post_reproj = [x[1] for x in reproj if x[0] == "post"]

    print("Creating pre mosaic")
    pre_mosaic = create_mosaic(pre_reproj, Path(f"{args.staging_directory}/mosaics/pre.tif"))
    print("Creating post mosaic")
    post_mosaic = create_mosaic(post_reproj, Path(f"{args.staging_directory}/mosaics/post.tif"))

    import ipdb; ipdb.set_trace()
    
    extent = get_intersect(pre_mosaic, post_mosaic)

    pre_chips = create_chips(pre_mosaic, args.output_directory.joinpath('chips').joinpath('pre'), extent)
    post_chips = create_chips(post_mosaic, args.output_directory.joinpath('chips').joinpath('post'), extent)
    
    """
    
    # FOR TESTING
    import os
    pre_mosaic = Path('/raid/data/jdunnmon/test/staging/mosaics/pre.tif')
    post_mosaic = Path('/raid/data/jdunnmon/test/staging/mosaics/post.tif')
    pre_chips = [Path(os.path.join('/raid/data/jdunnmon/test/out/chips/pre',a)) for a in os.listdir('/raid/data/jdunnmon/test/out/chips/pre')]
    post_chips = [Path(os.path.join('/raid/data/jdunnmon/test/out/chips/post',a)) for a in os.listdir('/raid/data/jdunnmon/test/out/chips/post')]
    
    # Loading config
    config = CfgNode.load_cfg(open(args.model_config_path, 'rb'))

    # Defining dataset and dataloader
    pairs = []
    for idx, (pre, post) in enumerate(zip(pre_chips, post_chips)):
        pairs.append(Files(
            pre.stem,
            args.pre_directory,
            args.post_directory,
            args.output_directory,
            pre,
            post)
            )

    eval_dataset = XViewDataset(pairs, config, transform=build_image_transforms())
    eval_dataloader = DataLoader(eval_dataset, batch_size=16, num_workers=8)

    # Loading model
    ckpt_path = args.model_weight_path
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    model = get_model(config)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['state_dict'])
    model.eval()

    model_wrapper = ModelWrapper(model, args.is_use_gpu, config.MODEL.IS_SPLIT_LOSS)
    model_wrapper.eval()
        
    # Running inference
    results = defaultdict(list)
    
    with torch.no_grad():
        for result_dict in tqdm(eval_dataloader, total=len(eval_dataloader)):
            loc, cls = model_wrapper(result_dict['pre_image'], result_dict['post_image'])
            loc = loc.detach().cpu().numpy().astype(np.uint8)
            cls = cls.detach().cpu().numpy().astype(np.uint8)
            
            result_dict['pre_image'] = result_dict['pre_image'].cpu().numpy()
            result_dict['post_image'] = result_dict['post_image'].cpu().numpy()
            result_dict['loc'] = loc
            result_dict['cls'] = cls
            result_dict['geo_profile'] = [eval_dataset.pairs[idx].opts['geo_profile']
                                          for idx in result_dict['idx']]
            for k,v in result_dict.items():
                results[k] = results[k] + list(v)
    
    import ipdb; ipdb.set_trace()
    # Running postprocessing
    p = mp.Pool(args.n_procs)
    f_p = partial(postprocess_and_write, config=config)
    p.starmap(f_p, results)
    
    # Complete
    print('Run complete!')
        
    
        


if __name__ == '__main__':
    main()
