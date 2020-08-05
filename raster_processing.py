import rasterio
import rasterio.merge
import rasterio.warp
from rasterio import windows
from itertools import product

import os


def reproject(in_file, dest_file, dest_crs='EPSG:4326'):

    with rasterio.open(in_file) as src:
        transform, width, height = rasterio.warp.calculate_default_transform(src.crs, dest_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'driver': 'GTiff',
            'crs': dest_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        with rasterio.open(dest_file, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dest_crs,
                    resampling=rasterio.warp.Resampling.nearest)

    return os.path.abspath(dest_file)


def create_mosaic(in_files, out_file='output/staging/mosaic.tif', plot=False):

    src_files = [rasterio.open(file) for file in in_files]

    mosaic, out_trans = rasterio.merge.merge(src_files)

    if plot:
        import rasterio.plot
        rasterio.plot.show(mosaic, cmap='terrain')

    out_meta = src_files[0].meta.copy()

    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans,
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
                                         abs(xy_ul[0] - xy_lr[0]),
                                         abs(xy_ul[1] - xy_lr[1]))

    return int_window


def create_chips(in_raster, out_dir, intersect):

    output_filename = 'tile_{}-{}.tif'

    def get_tiles(ds, width=1024, height=1024):
        #nols, nrows = ds.meta['width'], ds.meta['height']
        intersect_window = get_intersect_win(ds, intersect)
        offsets = product(range(intersect_window.col_off, intersect_window.width, width),
                          range(intersect_window.row_off, intersect_window.height, height))
        for col_off, row_off in offsets:
            window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height)
            transform = windows.transform(window, ds.transform)
            yield window, transform

    with rasterio.open(in_raster) as inds:
        tile_width, tile_height = 1024, 1024

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
