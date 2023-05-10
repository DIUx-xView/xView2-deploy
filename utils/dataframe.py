from pathlib import Path
from typing import Union

import geopandas
import rasterio
import rasterio.crs
import rasterio.warp
from affine import Affine
from loguru import logger
from shapely.geometry import Polygon


def make_footprint_df(files: list[Path]) -> geopandas.GeoDataFrame:
    """create footprint/metadata geodataframe from list of image files

    Args:
        files (list[Path]): list of path objects for input imagery

    Returns:
        geopandas.GeoDataFrame: geadataframe of footprints and select metadata
    """
    # Todo: this requires much more error handling and tests

    polys = []
    filename = []
    res = []
    im_crs = []
    bounds = []
    height = []
    width = []

    # Get values from source rasters
    for f in files:
        with rasterio.open(f) as src:
            # Create footprint polygon
            vert = [
                (src.bounds.left, src.bounds.top),
                (src.bounds.left, src.bounds.bottom),
                (src.bounds.right, src.bounds.bottom),
                (src.bounds.right, src.bounds.top),
            ]
            polys.append(Polygon(vert))

            filename.append(f)
            res.append(src.res)
            im_crs.append(src.crs)
            height.append(src.height)
            width.append(src.width)
            bounds.append(src.bounds)

    df = geopandas.GeoDataFrame(
        {
            "filename": filename,
            "crs": im_crs,
            "src_res": res,
            "height": height,
            "width": width,
            "bounds": bounds,
            "geometry": polys,
        }
    )

    # Set CRS by CRS of first image
    df.crs = df["crs"][0]

    return df


def df_from_file(in_file: str) -> geopandas.GeoDataFrame:
    """create GeoDataFrame from input file

    Args:
        in_file (str): Input file. Must be readable by GeoPandas (Fiona).

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame from file
    """

    return geopandas.GeoDataFrame.from_file(in_file)


def process_df(
    df: geopandas.GeoDataFrame, dest_crs: rasterio.crs.CRS
) -> geopandas.GeoDataFrame:
    """Adds "trans_res" column to GeoDataframe containing transformations from imagery

    Args:
        df (geopandas.GeoDataFrame): GeoDataframe of imagery items.
        dest_crs (rasterio.crs.CRS): Destination CRS as RasterIO object.

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame
    """
    df["trans_res"] = df.apply(
        lambda x: get_trans_res(x.crs, x.width, x.height, x.bounds, dest_crs), axis=1
    )
    # Todo: Remove rasters with resolution that wildly varies from the others
    # Todo: Remove rasters with CRS that is not the same as most?

    # Todo: handle our rasters with no geographic data
    # Remove rasters without transform resolution
    # df = df[df['trans_res'] is not None]

    # Remove rasters that do not have CRS set
    # df = df[df['crs'] is not None]

    return df


def get_utm(df: geopandas.GeoDataFrame) -> str:
    """Get UTM code for GeoDataframe. Dissolves all features and gets UTM at the centroid.

    The EPSG is calculates as:
        32600+zone for positive latitudes
        32700+zone for negatives latitudes

    Args:
        df (geopandas.GeoDataFrame): GeoDataframe

    Returns:
        rasterio.crs.CRS: RasterIO CRS object of UTM
    """

    cent = df.dissolve().geometry.to_crs(4326).centroid
    lon = cent.x[0]
    lat = cent.y[0]
    zone = int(round((183 + lon) / 6, 0))
    epsg = int(32700 - round((45 + lat) / 90, 0) * 100) + zone

    return rasterio.crs.CRS.from_string(f"EPSG:{epsg}")

    # Todo: This would be the preferred way, however it requires pyproj 3.0+ which seems to break every environment I tried.
    # return df.estimate_utm_crs()


def get_trans_res(
    src_crs: Union[rasterio.crs.CRS, dict],
    width: int,
    height: int,
    bounds: tuple[float],
    dst_crs: rasterio.crs.CRS,
) -> tuple[float]:
    """Calculates default transform resolution.
    :param src_crs: source CRS
    :param width: source width
    :param height: source height
    :param bounds: tuple of source bounds
    :param dst_crs: destination CRS
    :return: tuple of destination CRS resolution (x, y)
    Source coordinate reference system, in rasterio dict format.
    Example: CRS({'init': 'EPSG:4326'})


    Args:
        src_crs (Union[rasterio.crs.CRS, dict]): Source coordinate reference system, in rasterio dict format.
    Example: CRS({'init': 'EPSG:4326'})
        width (int): Image width
        height (int): Image height
        bounds (tuple[float]): Image bounds
        dst_crs (Union[rasterio.crs.CRS, dict]): Destination CRS

    Returns:
        tuple[float]: Transform resolution
    """
    # Get our transform
    transform = rasterio.warp.calculate_default_transform(
        src_crs,
        dst_crs,
        width=width,
        height=height,
        left=bounds.left,
        bottom=bounds.bottom,
        right=bounds.right,
        top=bounds.top,
        dst_width=width,
        dst_height=height,
    )

    return (transform[0][0], -transform[0][4])


def get_intersect(
    pre_df: geopandas.GeoDataFrame,
    post_df: geopandas.GeoDataFrame,
    args: object,
    aoi: geopandas.GeoDataFrame = None,
    in_poly_df: geopandas.GeoDataFrame = None,
) -> Polygon:
    """Computes intersection of two GeoDataFrames and reduces extent by an optional defined area of interest (AOI) and building footprint polygons.

    Args:
        pre_df (geopandas.GeoDataFrame): GeoDataFrame.
        post_df (geopandas.GeoDataFrame): GeoDataFrame.
        args (object): Arguments object.
        aoi (geopandas.GeoDataFrame, optional): AOI GeoDataframe. Defaults to None.
        in_poly_df (geopandas.GeoDataFrame, optional): Building footprint GeoDataFrame. Defaults to None.

    Returns:
        Polygon: Shapely polygon of intersection among all layers.
    """

    geom_bounds = []

    pre_env = pre_df.to_crs(args.destination_crs).unary_union
    geom_bounds.append(pre_env)
    logger.debug(f"Pre bounds: {pre_env.bounds}")

    post_env = post_df.to_crs(args.destination_crs).unary_union
    geom_bounds.append(post_env)
    logger.debug(f"Post bounds: {post_env.bounds}")

    intersect = pre_env.intersection(post_env)
    assert intersect.area > 0, logger.critical("Pre and post imagery do not intersect")

    if aoi is not None:
        aoi_env = aoi.to_crs(args.destination_crs).unary_union
        geom_bounds.append(aoi_env)
        logger.debug(f"AOI bounds: {aoi_env.bounds}")
        intersect = intersect.intersection(aoi_env)
        assert intersect.area > 0, logger.critical("AOI does not intersect imagery")

    if in_poly_df is not None:
        in_poly_env = in_poly_df.to_crs(args.destination_crs).unary_union
        geom_bounds.append(in_poly_env)
        logger.debug(f"In poly bounds: {in_poly_env.bounds}")
        intersect = intersect.intersection(in_poly_env)
        assert intersect.area > 0, logger.critical(
            "Building polygons do not intersect imagery/AOI"
        )

    return intersect


def get_max_res(
    pre_df: geopandas.GeoDataFrame, post_df: geopandas.GeoDataFrame
) -> tuple[float]:
    """Calculates maximum resolution from two GeoDataFrames of imagery footprints. This provides the maximum resolution that all imagery products can be reprojected to without upscaling.

    Args:
        pre_df (geopandas.GeoDataFrame): GeoDataframe of imagery footprints
        post_df (geopandas.GeoDataFrame): GeoDataframe of imagery footprints

    Returns:
        tuple[float]: Tuple of (x, y) resolutions.
    """

    res_list = list(pre_df.trans_res) + list(post_df.trans_res)
    x = max(x[0] for x in res_list)
    y = max(x[1] for x in res_list)
    return (x, y)


def bldg_poly_process(
    df: geopandas.GeoDataFrame,
    intersect: tuple[float],
    dest_crs: Union[rasterio.crs.CRS, dict],
    out_file: Union[Path, str],
    out_shape: tuple[int],
    transform: Affine,
) -> Path:
    """Process input polygon GeoDataFrame. Clips dataframe to 'intersect', reprojects to 'dest_crs', and rasterizes features.

    Args:
        df (geopandas.GeoDataFrame): Dataframe of input features
        intersect (tuple[float]): Tuple of calculated intersect
        dest_crs (Union[rasterio.crs.CRS, dict]): Destination CRS
        out_file (Union[Path, str]): Path or string of filepath for generated tiff
        out_shape (tuple[int]): Tuple of ('height', 'width')
        transform (Affine): Transform

    Returns:
        Path: _description_
    """
    def _clip_polys(input, mask):
        return geopandas.clip(input, mask)

    def _rasterize(in_feats, out_file, out_shape, transform, dst_crs):
        image = rasterio.features.rasterize(
            in_feats.geometry,
            out_shape=out_shape,
            all_touched=True,
            transform=transform,
        )

        assert image.sum() > 0, logger.critical(
            "Building polygon rasterization failed."
        )

        with rasterio.open(
            out_file,
            "w",
            driver="GTiff",
            dtype=rasterio.uint8,
            count=1,
            height=out_shape[0],
            width=out_shape[1],
            transform=transform,
            crs=dst_crs,
        ) as dst:
            dst.write(image, indexes=1)

        return out_file

    out_file = Path(out_file)

    poly_cords = [
        (intersect[0], intersect[1]),
        (intersect[2], intersect[1]),
        (intersect[2], intersect[3]),
        (intersect[0], intersect[3]),
    ]
    mask = geopandas.GeoDataFrame({"geometry": [Polygon(poly_cords)]}, crs=dest_crs)
    df = df.to_crs(dest_crs)

    assert mask.crs == df.crs, logger.critical("CRS mismatch")

    df = _clip_polys(df, mask)
    mosaic = _rasterize(df, out_file, out_shape, transform, dest_crs)

    return mosaic
