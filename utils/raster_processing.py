import io
from itertools import product
from pathlib import Path
from typing import Union

import numpy as np
import rasterio
import rasterio.crs
import rasterio.merge
import rasterio.plot
import rasterio.warp
from affine import Affine
from loguru import logger
from numpy import array
from osgeo import gdal, ogr
from PIL import Image
from rasterio import windows
from tqdm import tqdm


def get_res(image: Union[Path, str]) -> tuple[float]:
    """Get x,y resolution from raster

    Args:
        image (Union[Path, str]): Filename of raster.

    Returns:
        tuple[float]: Resolution of rastor (x, y)
    """
    with rasterio.open(image) as src:
        return src.res


def create_mosaic(
    in_data: list[Union[Path, str]],
    out_file: Union[Path, str],
    src_crs: rasterio.crs.CRS = None,
    dst_crs: rasterio.crs.CRS = None,
    extent: tuple[float] = None,
    dst_res: tuple[float] = None,
    aoi=None,
) -> Union[Path, str]:
    """Create mosaic from input files

    Args:
        in_data (Union[Path, str]): List of input rasters.
        out_file (Union[Path, str]): Filename to save mosaic
        src_crs (rasterio.crs.CRS, optional): Source CRS of rasters. Only used if CRS is not set on input rasters. Defaults to None.
        dst_crs (rasterio.crs.CRS, optional): Destination CRS of mosaic. Defaults to None.
        extent (tuple[float], optional): tuple of extent to clip mosaic to. Defaults to None.
        dst_res (tuple[float], optional): Destination mosaic resolution. Defaults to None.
        aoi (_type_, optional): _description_. Defaults to None.

    Returns:
        Union[Path, str]: _description_
    """

    # Note: gdal will not accept Path objects. They must be passed as strings
    if dst_res:
        xRes = (dst_res[0],)
        yRes = dst_res[1]
    else:
        xRes = None
        yRes = None

    if aoi is not None:
        if src_crs is not None:  # TODO: clean this up!
            aoi = aoi.to_crs(src_crs)
        else:
            raster = rasterio.open(in_data[0])
            aoi = aoi.to_crs(raster.crs)
            raster.close()

        with io.BytesIO() as f:
            aoi.to_file(f, driver="GeoJSON")
            f.seek(0)
            aoi = f.read().decode()

    temp_out = out_file.with_name(f"{out_file.stem}_temp.tif")
    # temp_out = out_file.with_stem(out_file.stem + '_temp') # Todo: Requires Python >= 3.9

    reproj = gdal.Warp(
        str(temp_out),
        in_data,
        srcSRS=src_crs,
        format="GTiff",
        dstSRS=dst_crs,
        xRes=xRes,
        yRes=yRes,
        outputBounds=extent,
        cutlineDSName=aoi,
        cropToCutline=True,  # Todo: Setting this to true creates huge files...the opposite of what I expect
        # Todo: Apparently crop will grew it the the size of the cutline dataset. So if is bigger than the raster it pads it
        multithread=True,
    )

    # Remove alpha channel
    gdal.Translate(str(out_file), reproj, bandList=[1, 2, 3])
    temp_out.unlink()

    return out_file


def check_dims(arr: array, w: int, h: int) -> array:
    """Check dimensions of output tiles and pad to create equal dimensions.

    Args:
        arr (array): Input numpy array.
        w (int): Width to pad to.
        h (int): Height to pad to.

    Returns:
        array: Input array padded to w, h values.
    """

    dims = arr.shape
    if dims[1] != w or dims[2] != h:
        result = np.zeros((arr.shape[0], w, h)).astype(arr.dtype)
        result[: arr.shape[0], : arr.shape[1], : arr.shape[2]] = arr
    else:
        result = arr

    return result


def create_chips(
    in_raster: Union[Path, str],
    out_dir: Path,
    intersect: tuple[float],
    tile_width: int = 1024,
    tile_height: int = 1024,
):
    """Create chips from raster that fall inside 'intersect'.

    Args:
        in_raster (Union[Path, str]): Input raster to create chips from.
        out_dir (Path): Output directory to save chips.
        intersect (tuple[float]): Intersect to constrain chips.
        tile_width (int, optional): Tile width. Defaults to 1024.
        tile_height (int, optional): Tile height. Defaults to 1024.
    """

    def get_intersect_win(rio_obj):
        """
        Calculate rasterio window from intersect
        :param rio_obj: rasterio dataset
        :return: window of intersect
        """

        xy_ul = rasterio.transform.rowcol(rio_obj.transform, intersect[0], intersect[3])
        xy_lr = rasterio.transform.rowcol(rio_obj.transform, intersect[2], intersect[1])

        int_window = rasterio.windows.Window(
            xy_ul[1], xy_ul[0], abs(xy_ul[1] - xy_lr[1]), abs(xy_ul[0] - xy_lr[0])
        )

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
        offsets = product(
            range(
                intersect_window.col_off,
                intersect_window.width + intersect_window.col_off,
                width,
            ),
            range(
                intersect_window.row_off,
                intersect_window.height + intersect_window.row_off,
                height,
            ),
        )
        for col_off, row_off in offsets:
            window = windows.Window(
                col_off=col_off, row_off=row_off, width=width, height=height
            ).intersection(intersect_window)
            transform = windows.transform(window, ds.transform)
            yield window, transform

    chips = []

    with rasterio.open(in_raster) as inds:
        meta = inds.meta.copy()

        for idx, (window, transform) in enumerate(
            tqdm(get_tiles(inds, tile_width, tile_height))
        ):
            meta["transform"] = transform
            meta["width"], meta["height"] = tile_width, tile_height
            output_filename = f"{idx}_{out_dir.parts[-1]}.tif"
            outpath = out_dir.joinpath(output_filename)

            with rasterio.open(outpath, "w", **meta) as outds:
                chip_arr = inds.read(window=window)
                out_arr = check_dims(chip_arr, tile_width, tile_height)
                assert out_arr.shape[1] == tile_width, logger.error(
                    "Tile dimensions not correct"
                )
                assert out_arr.shape[2] == tile_height, logger.error(
                    "Tile dimensions not correct"
                )

                outds.write(out_arr)

            chips.append(outpath.resolve())

    return chips


def create_composite(
    base: Union[Path, str],
    overlay: array,
    out_file: Union[Path, str],
    transforms: Affine,
    alpha: float = 0.6,
) -> Union[Path, str]:
    """Composites 'overlay' on top of 'base' with 'alpha'.

    Args:
        base (Union[Path, str]): Base image path.
        overlay (array): Numpy array to overlay on 'base'.
        out_file (Union[Path, str]): Path to save overlay.
        transforms (Affine): Affine transform to apply to saved overlay.
        alpha (float, optional): Desired alpha to apply to 'overlay'. Defaults to 0.6.

    Returns:
        Union[Path, str]: Overlay file path.
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
    no_alpha = comp_arr[:, :, :3]

    with rasterio.open(out_file, "w", **transforms) as dst:
        # Go from (x, y, bands) to (bands, x, y)
        no_alpha = np.flipud(no_alpha)
        no_alpha = np.rot90(no_alpha, 3)
        no_alpha = np.moveaxis(no_alpha, [0, 1, 2], [2, 1, 0])

        dst.write(no_alpha)

    return Path(out_file)
