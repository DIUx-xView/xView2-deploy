from itertools import product
import random
import resource
import string
import subprocess
import fiona
import numpy as np
import rasterio
import rasterio.merge
import rasterio.warp
import rasterio.plot
from rasterio import windows
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.geometry.polygon import Polygon
from shapely.ops import cascaded_union
from osgeo import gdal
from tqdm import tqdm
from handler import *
from pathlib import Path


def reproject(in_file, dest_file, in_crs, dest_crs='EPSG:4326'):

    """
    Re-project images
    :param in_file: path to file to be reprojected
    :param dest_file: path to write re-projected image
    :param in_crs: crs of input file -- only valid if image does not contain crs in metadata
    :param dest_crs: destination crs
    :return: path to re-projected image
    """

    input_raster = gdal.Open(str(in_file))

    if input_raster.GetSpatialRef() is not None:
        in_crs = input_raster.GetSpatialRef()

    if in_crs is None:
        raise ValueError('No CRS set')

    # TODO: Change the resolution based on the lowest resolution in the inputs
    gdal.Warp(str(dest_file), input_raster, dstSRS=dest_crs, srcSRS=in_crs, xRes=6e-06, yRes=6e-06)

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
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if len(in_files) >= soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (len(in_files) * 2, hard))

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

    # Reset soft limit
    if len(in_files) >= soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))

    return Path(out_file).resolve()


def get_intersect(*args):

    """
    Computes intersect of input rasters.
    :param args: list of files to compute
    :return: tuple of intersect in (left, bottom, right, top)
    """

    # TODO: Calculate real intersection.

    left = []
    bottom = []
    right = []
    top = []

    for arg in args:
        raster = rasterio.open(arg)
        left.append(raster.bounds[0])
        bottom.append(raster.bounds[1])
        right.append(raster.bounds[2])
        top.append(raster.bounds[3])

    intersect = (max(left), max(bottom), min(right), min(top))

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
                assert(out_arr.shape[1] == tile_width)
                assert(out_arr.shape[2] == tile_height)

                outds.write(out_arr)

            chips.append(outpath.resolve())

    return chips


def create_shapefile(in_files, out_shapefile, dest_crs):

    polygons = []
    for idx, f in enumerate(in_files):
        src = rasterio.open(f)
        crs = src.crs
        transform = src.transform

        bnd = src.read(1)
        polys = list(shapes(bnd, transform=transform))

        for geom, val in polys:
            if val == 0:
                continue
            polygons.append((Polygon(shape(geom)), val))

    shp_schema = {
        'geometry': 'Polygon',
        'properties': {'dmg': 'int'}
    }

    # Write out all the multipolygons to the same file
    with fiona.open(out_shapefile, 'w', 'ESRI Shapefile', shp_schema,
                    dest_crs) as shp:
        for polygon, px_val in polygons:
            shp.write({
                'geometry': mapping(polygon),
                'properties': {'dmg': int(px_val)}
            })


    ####

