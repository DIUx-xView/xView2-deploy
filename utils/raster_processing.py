import random
import string
import subprocess
import fiona
import numpy as np
import rasterio
import rasterio.merge
import rasterio.warp
import rasterio.plot
import rasterio.crs
import handler
import os
from rasterio import windows
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely.geometry.polygon import Polygon
from itertools import product
from osgeo import gdal
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from PIL import Image


def get_reproj_res(pre_files, post_files, args):

    def get_res(file, in_crs, dst_crs):

        with rasterio.open(file) as src:

            # See if the CRS is set in the file, else use the passed argument
            if src.crs:
                pre_crs = src.crs
            else:
                pre_crs = rasterio.crs.CRS({'init': in_crs})

            # Get our transform
            transform = rasterio.warp.calculate_default_transform(
                pre_crs,
                rasterio.crs.CRS({'init': dst_crs}),
                width=src.width, height=src.height,
                left=src.bounds.left,
                bottom=src.bounds.bottom,
                right=src.bounds.right,
                top=src.bounds.top,
                dst_width=src.width, dst_height=src.height
            )

            # Append resolution from the affine transform.
        return (transform[0][0], -transform[0][4])

    res = []

    for file in pre_files:
        # Try to skip non-geospatial images
        try:
            res.append(get_res(file, args.pre_crs, args.destination_crs))
        except AttributeError:
            pass

    for file in post_files:
        # Try to skip non-geospatial images
        try:
            res.append(get_res(file, args.post_crs, args.destination_crs))
        except AttributeError:
            pass

    return (max([sublist[0] for sublist in res]),
            max([sublist[1] for sublist in res]))


# Todo: This should be able to be skipped by passing the res to reproject.
def reproject(in_file, dest_file, in_crs, dest_crs, res):

    """
    Re-project images
    :param in_file: path to file to be reprojected
    :param dest_file: path to write re-projected image
    :param in_crs: crs of input file -- only valid if image does not contain crs in metadata
    :param dest_crs: destination crs
    :param res: tuple -- output resolution
    :return: path to re-projected image
    """

    input_raster = gdal.Open(str(in_file))

    if input_raster.GetSpatialRef() is not None:
        in_crs = input_raster.GetSpatialRef()

    if in_crs is None:
        raise ValueError('No CRS set')

    gdal.Warp(str(dest_file), input_raster, dstSRS=dest_crs, srcSRS=in_crs, xRes=res[0], yRes=res[1])

    return Path(dest_file).resolve()


def create_mosaic(in_files, out_file):

    """
    Creates mosaic from in_files.
    :param in_files: list of paths to input files
    :param out_file: path to output mosaic
    :return: path to output file
    """

    # This is some hacky, dumb shit
    # There is a limit on how many file descriptors we can have open at once
    # So we will up that limit for a bit and then set it back
    if os.name == 'posix':
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if len(in_files) >= soft:
            new_limit = len(in_files) * 2
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard))
            logger.debug(f'Hard limit changed from: {hard} to {new_limit}')
            logger.debug(f'Soft limit: {soft}')
    elif os.name == 'nt':
        import win32file
        soft = win32file._getmaxstdio()
        if len(in_files) >= soft:
            new_limit = len(in_files) * 2
            win32file._setmaxstdio(new_limit)
            logger.debug(f'Limit changed from {soft} to {new_limit}')

    file_objs = []

    for file in in_files:
        src = rasterio.open(file)
        file_objs.append(src)

    mosaic, out_trans = rasterio.merge.merge(file_objs)

    out_meta = src.meta.copy()

    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans
                     }
                    )

    with rasterio.open(out_file, "w", **out_meta) as dest:
        dest.write(mosaic)
        logger.debug(f'Mosaic created at {out_file} with resolution: {dest.res}')

    # Reset soft limit
    if os.name == 'posix':
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if len(in_files) >= soft:
            # Todo: this doesn't seem correct
            resource.setrlimit(resource.RLIMIT_NOFILE, (len(in_files) * 2, hard))
    elif os.name == 'nt':
        import win32file
        soft = win32file._getmaxstdio()
        if len(in_files) >= soft:
            win32file._setmaxstdio(len(in_files) * 2)
    # Todo: log limit reset

    return Path(out_file).resolve()


def get_intersect(pre_mosaic, post_mosaic):

    """
    Computes intersect of input two rasters.
    :param pre_mosaic: pre mosaic
    :param post_mosaic: post mosaic
    :return: tuple of intersect in (left, bottom, right, top)
    """

    with rasterio.open(pre_mosaic) as pre:
        pre_win = rasterio.windows.Window(0, 0, pre.width, pre.height)
        pre_bounds = pre.window_bounds(pre_win)
        logger.debug(f'Pre bounds: {pre_bounds}')

    with rasterio.open(post_mosaic) as post:
        post_win = rasterio.windows.Window(0, 0, post.width, post.height)
        pre_win_bounds = post.window(*pre_bounds)
        logger.debug(f'Post bounds: {post.window_bounds(post_win)}')
        assert rasterio.windows.intersect(pre_win_bounds, post_win), logger.error('Input rasters do not intersect')
        intersect_win = post_win.intersection(pre_win_bounds)
        intersect = post.window_bounds(intersect_win)
        logger.debug(f'Calculated intersect: {intersect}')

        return intersect


def check_dims(arr, w, h):
    """
    Check dimensions of output tiles and pad
    :param arr: numpy array
    :param w: tile width
    :param h: tile height
    :return: tile of same dimensions specified
    """

    dims = arr.shape
    if dims[1] != w or dims[2] != h:
        result = np.zeros((arr.shape[0],w,h)).astype(arr.dtype)
        result[:arr.shape[0],:arr.shape[1],:arr.shape[2]] = arr
    else:
        result = arr

    return result 


def create_chips(in_raster, out_dir, intersect, tile_width=1024, tile_height=1024):

    """
    Creates chips from mosaic that fall inside the intersect
    :param in_raster: mosaic to create chips from
    :param out_dir: path to write chips
    :param intersect: bounds of chips to create
    :param tile_width: width of tiles to chip
    :param tile_height: height of tiles to chip
    :return: list of path to chips
    """

    def get_intersect_win(rio_obj):

        """
        Calculate rasterio window from intersect
        :param rio_obj: rasterio dataset
        :return: window of intersect
        """

        xy_ul = rasterio.transform.rowcol(rio_obj.transform, intersect[0], intersect[3])
        xy_lr = rasterio.transform.rowcol(rio_obj.transform, intersect[2], intersect[1])

        int_window = rasterio.windows.Window(xy_ul[1], xy_ul[0],
                                             abs(xy_ul[1] - xy_lr[1]),
                                             abs(xy_ul[0] - xy_lr[0]))

        return int_window

    def get_tiles(ds, width, height):

        """
        Create chip tiles generator
        :param ds: rasterio dataset
        :param width: tile width
        :param height: tile height
        :return: generator of rasterio windows and transforms for each tile to be created
        """

        intersect_window = get_intersect_win(ds)
        offsets = product(range(intersect_window.col_off, intersect_window.width + intersect_window.col_off, width),
                          range(intersect_window.row_off, intersect_window.height + intersect_window.row_off, height))
        for col_off, row_off in offsets:
            window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(intersect_window)
            transform = windows.transform(window, ds.transform)
            yield window, transform

    chips = []

    with rasterio.open(in_raster) as inds:

        meta = inds.meta.copy()

        for idx, (window, transform) in enumerate(tqdm(get_tiles(inds, tile_width, tile_height))):
            meta['transform'] = transform
            meta['width'], meta['height'] = tile_width, tile_height
            output_filename = f'{idx}_{out_dir.parts[-1]}.tif'
            outpath = out_dir.joinpath(output_filename)

            with rasterio.open(outpath, 'w', **meta) as outds:
                chip_arr = inds.read(window=window)
                out_arr = check_dims(chip_arr, tile_width, tile_height)
                assert(out_arr.shape[1] == tile_width), logger.error('Tile dimensions not correct')
                assert(out_arr.shape[2] == tile_height), logger.error('Tile dimensions not correct')

                outds.write(out_arr)

            chips.append(outpath.resolve())

    return chips


def create_composite(base, overlay, out_file, transforms, alpha=.6):
    """
    Creates alpha composite on an image from a numpy array.
    :param base: Base image file
    :param overlay: Numpy array to overlay
    :param out_file: Destination file
    :param transforms: Geo profile
    :param alpha: Desired alpha
    :return: Path object to overlay
    """

    # Todo: This seems pretty inefficient...would be a good candidate for refactoring.
    mask_map_img = np.zeros((overlay.shape[0], overlay.shape[1], 4), dtype=np.uint8)
    mask_map_img[overlay == 1] = (255, 255, 255, 255 * alpha)
    mask_map_img[overlay == 2] = (229, 255, 50, 255 * alpha)
    mask_map_img[overlay == 3] = (255, 159, 0, 255 * alpha)
    mask_map_img[overlay == 4] = (255, 0, 0, 255 * alpha)

    over_img = Image.fromarray(mask_map_img)
    pre_img = Image.open(base)
    pre_img.putalpha(255)

    comp = Image.alpha_composite(pre_img, over_img)
    comp_arr = np.asarray(comp)
    no_alpha = comp_arr[:,:,:3]

    with rasterio.open(out_file, 'w', **transforms) as dst:
        # Go from (x, y, bands) to (bands, x, y)
        no_alpha = np.flipud(no_alpha)
        no_alpha = np.rot90(no_alpha, 3)
        no_alpha = np.moveaxis(no_alpha, [0, 1, 2], [2, 1, 0])

        dst.write(no_alpha)

    return Path(out_file)
