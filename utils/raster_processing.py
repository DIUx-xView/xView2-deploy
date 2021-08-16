import numpy as np
import rasterio
import rasterio.merge
import rasterio.warp
import rasterio.plot
import rasterio.crs
from osgeo import gdal
from rasterio import windows
from itertools import product
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from PIL import Image


def create_vrt(in_files, out_path, resolution='lowest'):
    # Note: gdal does not accept path objects

    files = [str(file) for file in in_files]
    out_file = str(out_path)
    vrt = gdal.BuildVRT(out_file, files, resolution=resolution)
    return vrt


def get_res(image):
    with rasterio.open(image) as src:
        return src.res


def create_mosaic(in_data, in_vrt, out_file, dst_crs, dst_res, src_crs):
    # Note: gdal will not accept Path objects. They must be passed as strings

    reproj = gdal.Warp(str(out_file), in_files, format='GTiff', dstSRS=dst_crs, xRes=dst_res[0], yRes=dst_res[1])
    return out_file



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
