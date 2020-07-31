import rasterio
import rasterio.merge
import rasterio.plot
import subprocess
import os


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
                     # TODO: Update our crs
                     "crs": "+proj=utm +zone=15 +ellps=GRS80 +units=m +no_defs "
                     }
                    )

    with rasterio.open(out_file, "w", **out_meta) as dest:
        dest.write(mosaic)

    return os.path.abspath(out_file)
