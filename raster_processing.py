import rasterasterio
import rasterasterio.merge
import rasterasterio.plot
import rasterasterio.warp
from rasterasterio import windows
from itertools import product

import os


def reproject(in_file, dest_file, dest_crs='EPSG:4326'):

    with rasterasterio.open(in_file) as src:
        transform, width, height = rasterasterio.warp.calculate_default_transform(src.crs, dest_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'driver': 'GTiff',
            'crs': dest_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        with rasterasterio.open(dest_file, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                rasterasterio.warp.reproject(
                    source=rasterasterio.band(src, i),
                    destination=rasterasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dest_crs,
                    resampling=rasterasterio.warp.Resampling.nearest)

    return os.path.abspath(dest_file)


def create_mosaic(in_files, out_file='output/staging/mosaic.tif'):

    src_files = [rasterasterio.open(file) for file in in_files]

    mosaic, out_trans = rasterasterio.merge.merge(src_files)

    # TODO: make this an option
    rasterasterio.plot.show(mosaic, cmap='terrain')

    out_meta = src_files[0].meta.copy()

    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans,
                     }
                    )

    with rasterasterio.open(out_file, "w", **out_meta) as dest:
        dest.write(mosaic)

    return os.path.abspath(out_file)


def get_intersect(*args):
    """

    :param args:
    :return: Tuple of intersect extent in (left, bottom, right, top, (resx, resy))
    """
    # TODO: This has been tested for NW hemisphere. Real intersection would be ideal.

    left = []
    bottom = []
    right = []
    top = []
    resx = []
    resy = []

    for arg in args:
        raster = rasterasterio.open(arg)
        left.append(raster.bounds[0])
        bottom.append(raster.bounds[1])
        right.append(raster.bounds[2])
        top.append(raster.bounds[3])
        resx.append(raster.res[0])
        resy.append(raster.res[1])

    intersect = (max(left), max(bottom), min(right), min(top), (max(resx), max(resy)))

    return intersect


def create_chips(in_raster, out_dir):

    output_filename = 'tile_{}-{}.tif'

    def get_tiles(ds, width=256, height=256):
        nols, nrows = ds.meta['width'], ds.meta['height']
        offsets = product(range(0, nols, width), range(0, nrows, height))
        big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
        for col_off, row_off in  offsets:
            window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
            transform = windows.transform(window, ds.transform)
            yield window, transform


    with rasterio.open(in_raster) as inds:
        tile_width, tile_height = 256, 256

        meta = inds.meta.copy()
        print(meta)

        for window, transform in get_tiles(inds):
            print(window)
            meta['transform'] = transform
            meta['width'], meta['height'] = window.width, window.height
            outpath = os.path.join(out_dir,output_filename.format(int(window.col_off), int(window.row_off)))
            with rasterio.open(outpath, 'w', **meta) as outds:
                print(outds.meta)
                outds.write(inds.read(window=window))
