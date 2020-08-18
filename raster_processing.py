import rasterio
import rasterio.merge
import rasterio.warp
import rasterio.plot
from rasterio import windows
from itertools import product
from osgeo import gdal
import os


def reproject(in_file, dest_file, in_crs, dest_crs='EPSG:4326'):

    input_raster = gdal.Open(in_file)

    if input_raster.GetSpatialRef() is not None:
        in_crs = input_raster.GetSpatialRef()

    if in_crs is None:
        raise Exception('No CRS set')

    # TODO: Change the resolution based on the lowest resolution in the inputs
    gdal.Warp(dest_file, input_raster, dstSRS=dest_crs, srcSRS=in_crs, xRes=6e-06, yRes=6e-06)

    return os.path.abspath(dest_file)


def create_mosaic(in_files, out_file='output/staging/mosaic.tif'):

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

    return os.path.abspath(out_file)


def get_intersect(*args):
    """

    :param args:
    :return: Tuple of intersect extent in (left, bottom, right, top)
    """
    # TODO: This has been tested for NW hemisphere. Real intersection would be ideal.

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


def get_intersect_win(rio_obj, intersect):

    xy_ul = rasterio.transform.rowcol(rio_obj.transform, intersect[0], intersect[3])
    xy_lr = rasterio.transform.rowcol(rio_obj.transform, intersect[2], intersect[1])

    int_window = rasterio.windows.Window(xy_ul[1], xy_ul[0],
                                         abs(xy_ul[1] - xy_lr[1]),
                                         abs(xy_ul[0] - xy_lr[0]))

    return int_window


def create_chips(in_raster, out_dir, intersect):
    output_filename = 'tile_{}-{}.tif'

    def get_tiles(ds, width=1024, height=1024):
        #nols, nrows = ds.meta['width'], ds.meta['height']
        intersect_window = get_intersect_win(ds, intersect)
        offsets = product(range(intersect_window.col_off, intersect_window.width + intersect_window.col_off, width),
                          range(intersect_window.row_off, intersect_window.height + intersect_window.row_off, height))
        for col_off, row_off in offsets:
            window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(intersect_window)
            transform = windows.transform(window, ds.transform)
            yield window, transform


    with rasterio.open(in_raster) as inds:
        tile_width, tile_height = 1024, 1024

        meta = inds.meta.copy()

        for window, transform in get_tiles(inds):
            meta['transform'] = transform
            meta['width'], meta['height'] = window.width, window.height
            outpath = os.path.join(out_dir,output_filename.format(int(window.col_off), int(window.row_off)))

            with rasterio.open(outpath, 'w', **meta) as outds:
                outds.write(inds.read(window=window))
