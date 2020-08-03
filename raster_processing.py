import rasterio
import rasterio.merge
import rasterio.plot
import rasterio.warp
import subprocess
import os


def reproject(in_file, dest_file, dest_crs='EPSG:4326'):

    with rasterio.open(in_file) as src:
        transform, width, height = rasterio.warp.calculate_default_transform(src.crs, dest_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'driver': src.driver,
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

    return True


def create_mosaic(in_files, out_file='output/staging/mosaic.tif'):

    src_files = [rasterio.open(file) for file in in_files]

    mosaic, out_trans = rasterio.merge.merge(src_files)

    rasterio.plot.show(mosaic, cmap='terrain')

    # TODO: Add in metadata: https://automating-gis-processes.github.io/CSC18/lessons/L6/raster-mosaic.html
    out_meta = src_files[0].meta.copy()

    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans,
                     # TODO: Should be in ESPG 4326 due to reprojecting input files.
                     #"crs": "+proj=latlong"
                     }
                    )

    with rasterio.open(out_file, "w", **out_meta) as dest:
        dest.write(mosaic)

    return os.path.abspath(out_file)


def get_intersect(*args):

    rasterio.open(args[0])


def create_chips(in_raster, out_dir):
    pass