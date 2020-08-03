import rasterio
import rasterio.merge
import rasterio.plot
import rasterio.warp
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


def create_mosaic(in_files, out_file='output/staging/mosaic.tif'):

    src_files = [rasterio.open(file) for file in in_files]

    mosaic, out_trans = rasterio.merge.merge(src_files)

    # TODO: make this an option
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
        raster = rasterio.open(arg)
        left.append(raster.bounds[0])
        bottom.append(raster.bounds[1])
        right.append(raster.bounds[2])
        top.append(raster.bounds[3])
        resx.append(raster.res[0])
        resy.append(raster.res[1])

    intersect = (max(left), max(bottom), min(right), min(top), (max(resx), max(resy)))

    return intersect


def create_chips(in_raster, out_dir):
    # https://gis.stackexchange.com/questions/367832/using-rasterio-to-crop-image-using-pixel-coordinates-instead-of-geographic-coord
    pass