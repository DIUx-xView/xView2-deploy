import rasterio
import rasterio.merge
import rasterio.plot
import subprocess
import os


def create_mosaic(path, in_files, out_file='output/staging/mosaic.tif'):

    src_files = []

    for file in in_files:
        src = rasterio.open(file)
        src_files.append(src)

    mosaic, out_trans = rasterio.merge.merge(src_files)

    rasterio.plot.show(mosaic, cmap='terrain')

    out_meta = src.meta.copy()

    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans,
                     "crs": "+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs "
                     }
                    )

    with rasterio.open(out_file, "w", **out_meta) as dest:
        dest.write(mosaic)

    return os.path.abspath(out_file)