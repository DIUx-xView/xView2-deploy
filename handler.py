import cv2
import timeit
import glob
import argparse
import os
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
import sys
import resource
import numpy as np
import raster_processing
import rasterio.warp
import torch
#import ray
from collections import defaultdict
from os import makedirs, path
from functools import partial
from pathlib import Path
from shapely.geometry import mapping
from torch.utils.data import DataLoader
from yacs.config import CfgNode
from skimage.morphology import square, dilation
from tqdm import tqdm


from dataset import XViewDataset
from models import XViewFirstPlaceLocModel, XViewFirstPlaceClsModel

# TODO: Clean up directory structure
# TODO: gather input and output files from folders --> create pre and post mosaic --> create intersection --> get chips from intersection for pre/post --> extract geotransform per chip --> hand off to inference --> georef outputs

import functools
import logging
import struct
import sys

logger = logging.getLogger()

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
        self.opts = Options(pre_path=self.pre,
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

    # TODO: Does this method of making directories work on windows or do we need to use .joinpath?
    Path(f"{staging_path}/pre").mkdir(parents=True, exist_ok=True)
    Path(f"{staging_path}/post").mkdir(parents=True, exist_ok=True)


    return True


def make_output_structure(output_path):

    """
    Creates directory structure for outputs.
    :param output_path: Output path
    :return: True if succussful
    """

    Path(f"{output_path}/mosaics").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/chips/pre").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/chips/post").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/loc").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/dmg").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/over").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/shapes").mkdir(parents=True, exist_ok=True)

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
    """
    Helper function for reprojection
    """
    (pre_post, src_crs, raster_file) = raster_tuple
    basename = raster_file.stem
    dest_file = args.staging_directory.joinpath('pre').joinpath(f'{basename}.tif')
    try:
        return_dict[procnum] = (pre_post, raster_processing.reproject(raster_file, dest_file, src_crs, args.destination_crs))
    except ValueError:
        return None


def postprocess_and_write(result_dict):
    """
    Postprocess results from inference and write results to file
    :param result_dict: dictionary containing all required opts for each example
    """
    _thr = [0.38, 0.13, 0.14]
    pred_coefs = [1.0] * 4 # not 12, b/c already took mean over 3 in each subset 
    loc_coefs = [1.0] * 4 

    preds = []
    _i = -1
    for k,v in result_dict.items():
        if 'cls' in k:
            _i += 1
            # I think the below can just be replaced by v['cls'] -- shoul dcheck
            msk = v['cls'].numpy()
            preds.append(msk * pred_coefs[_i])
    
    preds = np.asarray(preds).astype('float').sum(axis=0) / np.sum(pred_coefs) / 255
    
    loc_preds = []
    _i = -1
    for k,v in result_dict.items():
        if 'loc' in k:
            _i += 1
            msk = v['loc'].numpy()
            loc_preds.append(msk * loc_coefs[_i])
    
    loc_preds = np.asarray(loc_preds).astype('float').sum(axis=0) / np.sum(loc_coefs) / 255
    
    msk_dmg = preds[..., 1:].argmax(axis=2) + 1
    msk_loc = (1 * ((loc_preds > _thr[0]) | ((loc_preds > _thr[1]) & (msk_dmg > 1) & (msk_dmg < 4)) | ((loc_preds > _thr[2]) & (msk_dmg > 1)))).astype('uint8')
    
    msk_dmg = msk_dmg * msk_loc
    _msk = (msk_dmg == 2)
    if _msk.sum() > 0:
        _msk = dilation(_msk, square(5))
        msk_dmg[_msk & msk_dmg == 1] = 2

    msk_dmg = msk_dmg.astype('uint8')

    loc = msk_loc
    cls = msk_dmg
    
    sample_result_dict = result_dict['34loc']
    sample_result_dict['geo_profile'].update(dtype=rasterio.uint8)

    with rasterio.open(sample_result_dict['out_loc_path'], 'w', **sample_result_dict['geo_profile']) as dst:
        dst.write(loc, 1)

    with rasterio.open(sample_result_dict['out_cls_path'], 'w', **sample_result_dict['geo_profile']) as dst:
        dst.write(cls, 1)
        
    if sample_result_dict['is_vis']:
        #TODO: Make sure this works with First Place code!
        mask_map_img = np.zeros((cls.shape[0], cls.shape[1], 3), dtype=np.uint8)
        mask_map_img[cls == 1] = (255, 255, 255)
        mask_map_img[cls == 2] = (229, 255, 50)
        mask_map_img[cls == 3] = (255, 159, 0)
        mask_map_img[cls == 4] = (255, 0, 0)

        # debug
        # cv2.imwrite('test_map.png',mask_map_img,[cv2.IMWRITE_PNG_COMPRESSION, 9])
        
        out_dir = os.path.dirname(sample_result_dict['out_overlay_path'])
        with rasterio.open(sample_result_dict['out_overlay_path'], 'w', **sample_result_dict['geo_profile']) as dst:
            # Go from (x, y, bands) to (bands, x, y)
            mask_map_img = np.flipud(mask_map_img)
            mask_map_img = np.rot90(mask_map_img, 3)
            mask_map_img = np.moveaxis(mask_map_img, [0, 1, 2], [2, 1, 0])
            dst.write(mask_map_img)

def run_inference(loader, model_wrapper, write_output=False, mode='loc', return_dict=None):
    results = defaultdict(list)
    pred_folder = model_wrapper.pred_folder
    with torch.no_grad(): # This is really important to not explode memory with gradients!
        for ii, result_dict in tqdm(enumerate(loader), total=len(loader)):
            #print(result_dict['in_pre_path'])
            debug=False
            #if '116' in result_dict['in_pre_path'][0]:
            #    import ipdb; ipdb.set_trace()
            #    debug=True
            out = model_wrapper.forward(result_dict['img'],debug=debug)
            out = out.detach().cpu()
            
            del result_dict['img']

            if 'pre_image' in result_dict:
                result_dict['pre_image'] = result_dict['pre_image'].cpu().numpy()
            if 'post_img' in result_dict:
                result_dict['post_image'] = result_dict['post_image'].cpu().numpy()
            if mode == 'loc':
                result_dict['loc'] = out
            elif mode == 'cls':
                result_dict['cls'] = out
            else:
                raise ValueError('Incorrect mode -- must be loc or cls')
            # Do this one separately because you can't return a class from a dataloader
            result_dict['geo_profile'] = [loader.dataset.pairs[idx].opts.geo_profile
                                          for idx in result_dict['idx']]
            for k,v in result_dict.items():
                results[k] = results[k] + list(v)
                
    # Making a list
    results_list = [dict(zip(results,t)) for t in zip(*results.values())]
    if write_output:
        print('Writing results...')
        makedirs(pred_folder, exist_ok=True)
        for result in tqdm(results_list, total=len(results_list)):
            # TODO: Multithread this to make it more efficient/maybe eliminate it from workflow
            if mode == 'loc':
                cv2.imwrite(path.join(pred_folder, 
                                  result['in_pre_path'].split('/')[-1].replace('.tif', '_part1.png')),
                                   np.array(result['loc'])[...], 
                                   [cv2.IMWRITE_PNG_COMPRESSION, 9])
            elif mode == 'cls':
                cv2.imwrite(path.join(pred_folder, result['in_pre_path'].split('/')[-1].replace('.tif', '_part1.png')),
                                      np.array(result['cls'])[..., :3], [cv2.IMWRITE_PNG_COMPRESSION, 9])
                cv2.imwrite(path.join(pred_folder, result['in_pre_path'].split('/')[-1].replace('.tif', '_part2.png')),
                                      np.array(result['cls'])[..., 2:], [cv2.IMWRITE_PNG_COMPRESSION, 9])    
    if return_dict is None:
        return results_list
    else:
        return_dict[f'{model_wrapper.model_size}{mode}'] = results_list

def main():
    
    t0 = timeit.default_timer()

    parser = argparse.ArgumentParser(description='Create arguments for xView 2 handler.')

    parser.add_argument('--pre_directory', metavar='/path/to/pre/files/', type=Path, required=True)
    parser.add_argument('--post_directory', metavar='/path/to/post/files/', type=Path, required=True)
    parser.add_argument('--staging_directory', metavar='/path/to/staging/', type=Path, required=True)
    parser.add_argument('--output_directory', metavar='/path/to/output/', type=Path, required=True)
    parser.add_argument('--model_weight_path', metavar='/path/to/model/weights', type=Path)
    parser.add_argument('--model_config_path', metavar='/path/to/model/config', type=Path)
    parser.add_argument('--is_use_gpu', action='store_true', help="If True, use GPUs")
    parser.add_argument('--n_procs', default=4, help="Number of processors for multiprocessing", type=int)
    parser.add_argument('--batch_size', default=16, help="Number of chips to run inference on at once", type=int)
    parser.add_argument('--num_workers', default=8, help="Number of workers loading data into RAM. Recommend 4 * num_gpu", type=int)
    parser.add_argument('--pre_crs', help='The Coordinate Reference System (CRS) for the pre-disaster imagery.')
    parser.add_argument('--post_crs', help='The Coordinate Reference System (CRS) for the post-disaster imagery.')
    parser.add_argument('--destination_crs', default='EPSG:4326', help='The Coordinate Reference System (CRS) for the output overlays.')
    parser.add_argument('--create_overlay_mosaic', default=False, action='store_true', help='True/False to create a mosaic out of the overlays')
    parser.add_argument('--create_shapefile', default=False, action='store_true', help='True/False to create shapefile from damage overlay')
    parser.add_argument('--dp_mode', default=False, action='store_true', help='True/False to run models serially, but using DataParallel')
    parser.add_argument('--save_intermediates', default=False, action='store_true', help='True/False to store intermediate runfiles')


    args = parser.parse_args()

    make_staging_structure(args.staging_directory)
    make_output_structure(args.output_directory)

    print('Retrieving files...')
    pre_files = get_files(args.pre_directory)
    post_files = get_files(args.post_directory)

    print('Re-projecting...')

    # Run reprojection in parallel processes
    manager = mp.Manager()
    return_dict = manager.dict()
    jobs = []

    # Some data hacking to make it more efficient for multiprocessing
    pre_files = [("pre", args.pre_crs, x) for x in pre_files]
    post_files = [("post", args.post_crs, x) for x in post_files]
    files = pre_files + post_files

    # Launch multiprocessing jobs for reprojection
    for idx, f in enumerate(files):
        p = mp.Process(target=reproject_helper, args=(args, f, idx, return_dict))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()

    reproj = [x for x in return_dict.values() if x[1] is not None]
    pre_reproj = [x[1] for x in reproj if x[0] == "pre"]
    post_reproj = [x[1] for x in reproj if x[0] == "post"]

    print("Creating pre mosaic...")
    pre_mosaic = raster_processing.create_mosaic(pre_reproj, Path(f"{args.output_directory}/mosaics/pre.tif"))
    print("Creating post mosaic...")
    post_mosaic = raster_processing.create_mosaic(post_reproj, Path(f"{args.output_directory}/mosaics/post.tif"))

    extent = raster_processing.get_intersect(pre_mosaic, post_mosaic)

    print('Chipping...')
    pre_chips = raster_processing.create_chips(pre_mosaic, args.output_directory.joinpath('chips').joinpath('pre'), extent)
    post_chips = raster_processing.create_chips(post_mosaic, args.output_directory.joinpath('chips').joinpath('post'), extent)

    # debug
    #pre_chips =  [bb for bb in pre_chips if '116' in str(bb)]
    #post_chips =  [bb for bb in post_chips if '116' in str(bb)]
    
    assert len(pre_chips) == len(post_chips)

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
    
    eval_loc_dataset = XViewDataset(pairs, 'loc')
    eval_loc_dataloader = DataLoader(eval_loc_dataset, 
                                     batch_size=args.batch_size, 
                                     num_workers=args.num_workers,
                                     shuffle=False,
                                     pin_memory=True)
    
    eval_cls_dataset = XViewDataset(pairs, 'cls')
    eval_cls_dataloader = DataLoader(eval_cls_dataset, 
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     shuffle=False,
                                     pin_memory=True)
    
    
    if args.dp_mode:
        results_dict = {}
        
        for sz in ['34', '50', '92', '154']:
            print(f'Running models of size {sz}...')
            return_dict = {}
            loc_wrapper = XViewFirstPlaceLocModel(sz, dp_mode=args.dp_mode)
            
            run_inference(eval_loc_dataloader,
                                loc_wrapper,
                                args.save_intermediates,
                                'loc',
                                return_dict)
            
            del loc_wrapper
            
            cls_wrapper = XViewFirstPlaceClsModel(sz, dp_mode=args.dp_mode)
            
            run_inference(eval_cls_dataloader,
                                cls_wrapper,
                                args.save_intermediates,
                                'cls',
                                return_dict)
            
            del cls_wrapper
            
            results_dict.update({k:v for k,v in return_dict.items()})
        
        
    elif torch.cuda.device_count() == 2:
        # For 2-GPU machines [TESTED]
        
        # Loading model
        loc_gpus = {'34':[0,0,0],
                    '50':[1,1,1],
                    '92':[0,0,0],
                    '154':[1,1,1]}

        cls_gpus = {'34':[1,1,1],
                    '50':[0,0,0],
                    '92':[1,1,1],
                    '154':[0,0,0]}

        results_dict = {}
        
        # Running inference
        print('Running inference...')
        
        for sz in loc_gpus.keys():
            print(f'Running models of size {sz}...')
            loc_wrapper = XViewFirstPlaceLocModel(sz, devices=loc_gpus[sz])
            cls_wrapper = XViewFirstPlaceClsModel(sz, devices=cls_gpus[sz])

            # Running inference
            print('Running inference...')

            # Run inference in parallel processes
            manager = mp.Manager()
            return_dict = manager.dict()
            jobs = []

            # Launch multiprocessing jobs for different pytorch jobs
            p1 = mp.Process(target=run_inference,
                            args=(eval_cls_dataloader,
                                cls_wrapper,
                                args.save_intermediates,
                                'cls',
                                return_dict))
            p2 = mp.Process(target=run_inference,
                            args=(eval_loc_dataloader,
                                loc_wrapper,
                                args.save_intermediates,
                                'loc',
                                return_dict))
            p1.start()
            p2.start()
            jobs.append(p1)
            jobs.append(p2)
            for proc in jobs:
                proc.join()

            results_dict.update({k:v for k,v in return_dict.items()})
            
    elif torch.cuda.device_count() == 8:
        # For 8-GPU machines
        # TODO: Test!
        
        # Loading model
        loc_gpus = {'34':[0,0,0],
                    '50':[1,1,1],
                    '92':[2,2,2],
                    '154':[3,3,3]}

        cls_gpus = {'34':[4,4,4],
                    '50':[5,5,5],
                    '92':[6,6,6],
                    '154':[7,7,7]}

        results_dict = {}
         # Run inference in parallel processes
        manager = mp.Manager()
        return_dict = manager.dict()
        jobs = []
        
        for sz in loc_gpus.keys():
            print(f'Adding jobs for size {sz}...')
            loc_wrapper = XViewFirstPlaceLocModel(sz, devices=loc_gpus[sz])
            cls_wrapper = XViewFirstPlaceClsModel(sz, devices=cls_gpus[sz])
            
            # DEBUG
            #run_inference(eval_loc_dataloader,
            #                    loc_wrapper,
            #                    True, # Don't write intermediate outputs
            #                    'loc',
            #                    return_dict)
            
            #import ipdb; ipdb.set_trace()

            # Launch multiprocessing jobs for different pytorch jobs
            jobs.append(mp.Process(target=run_inference,
                            args=(eval_cls_dataloader,
                                cls_wrapper,
                                args.save_intermediates, # Don't write intermediate outputs
                                'cls',
                                return_dict))
                            )
            jobs.append(mp.Process(target=run_inference,
                            args=(eval_loc_dataloader,
                                loc_wrapper,
                                args.save_intermediates, # Don't write intermediate outputs
                                'loc',
                                return_dict))
                            )

        print('Running inference...')

        for proc in jobs:
            proc.start()
        for proc in jobs:
            proc.join()

        results_dict.update({k:v for k,v in return_dict.items()})
    
    else:
        raise ValueError('Must use either 2 or 8 GPUs')
       
    # Quick check to make sure the samples in cls and loc are in teh same orer
    #assert(results_dict['34loc'][4]['in_pre_path'] == results_dict['34cls'][4]['in_pre_path'])
    
    results_list = [{k:v[i] for k,v in results_dict.items()} for i in range(len(results_dict['34cls'])) ]

    # Running postprocessing
    p = mp.Pool(args.n_procs)
    #postprocess_and_write(results_list[0])
    f_p = postprocess_and_write
    p.map(f_p, results_list)
    
    if args.create_overlay_mosaic:
        print("Creating overlay mosaic")
        p = Path(args.output_directory) / "over"
        overlay_files = get_files(p)
        overlay_files = [x for x in overlay_files]
        overlay_mosaic = raster_processing.create_mosaic(overlay_files, Path(f"{args.output_directory}/mosaics/overlay.tif"))

    if args.create_shapefile:
        print('Creating shapefile')
        files = get_files(Path(args.output_directory) / 'dmg')
        raster_processing.create_shapefile(files,
                         Path(args.output_directory).joinpath('shapes') / 'damage.shp',
                         args.destination_crs)

    # Complete
    print('Run complete!')
    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))

if __name__ == '__main__':
    if os.name == 'nt':
        from multiprocessing import freeze_support
        freeze_support()
    main()
