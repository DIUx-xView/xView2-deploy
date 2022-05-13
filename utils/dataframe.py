import geopandas
import rasterio
import rasterio.warp
import rasterio.crs
from shapely.geometry import Polygon
from loguru import logger


def make_footprint_df(files):
    """
    Creates a dataframe of raster footprints and metadata
    :param files: iterable of roster filenames
    :return: geodataframe of raster footprints and metadata
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
                (src.bounds.right, src.bounds.top)
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
            'filename': filename,
            'crs': im_crs,
            'src_res': res,
            'height': height,
            'width': width,
            'bounds': bounds,
            'geometry': polys
        }
    )

    # Set CRS by CRS of first image
    df.crs = df['crs'][0]

    return df


def make_aoi_df(aoi_file):
    aoi = {'wildfire': 'https://opendata.arcgis.com/datasets/2191f997056547bd9dc530ab9866ab61_0.geojson'}

    if aoi_file is None:
        return None

    if aoi_file in aoi:
        aoi_file = aoi.get(aoi_file)

    return geopandas.GeoDataFrame.from_file(aoi_file)


def process_df(df, dest_crs):
    """
    Process geadataframe of raster footprints and adds tranform resolution
    :param df: geodataframe
    :param dest_crs: destination CRS to get transform resolution
    :return: geodataframe
    """
    df['trans_res'] = df.apply(lambda x: get_trans_res(x.crs, x.width, x.height, x.bounds, dest_crs), axis=1)
    # Remove rasters with resolution that wildly varies from the others
    # Remove rasters with CRS that is not the same as most?

    # Todo: handle our rasters with no geographic data
    # Remove rasters without transform resolution
    #df = df[df['trans_res'] is not None]

    # Remove rasters that do not have CRS set
    #df = df[df['crs'] is not None]

    return df


def get_utm(df):
    """
    Calculate UTM EPSG code for coordinate pair
        The EPSG is:
        32600+zone for positive latitudes
        32700+zone for negatives latitudes
    :param df: geodataframe
    :return: EPSG code
    """
    """Return UTM EPSG code of respective lon/lat.

    """

    cent = df.dissolve().geometry.to_crs(4326).centroid
    lon = cent.x[0]
    lat = cent.y[0]
    zone = int(round((183 + lon) / 6, 0))
    epsg = int(32700 - round((45 + lat) / 90, 0) * 100) + zone

    return rasterio.crs.CRS.from_string(f'EPSG:{epsg}')

    # Todo: This would be the preferred way, however it requires pyproj 3.0+ which seems to break every environment I tried.
    #return df.estimate_utm_crs()


def get_trans_res(src_crs, width, height, bounds, dst_crs):
    """
    Calculates default transform resolution.
    :param src_crs: source CRS
    :param width: source width
    :param height: source height
    :param bounds: tuple of source bounds
    :param dst_crs: destination CRS
    :return: tuple of destination CRS resolution (x, y)
    """
    # Get our transform
    transform = rasterio.warp.calculate_default_transform(
        src_crs,
        dst_crs,
        width=width, height=height,
        left=bounds.left,
        bottom=bounds.bottom,
        right=bounds.right,
        top=bounds.top,
        dst_width=width, dst_height=height
    )

    return (transform[0][0], -transform[0][4])


def get_intersect(pre_df, post_df, args, aoi=None, in_poly_df=None):
    """
    Computes intersection of two dataframes and reduces extent by an optional defined AOI.
    :param pre_df: dataframe of raster footprints
    :param post_df: dataframe of raster footprints
    :param args: arguments object
    :param aoi: AOI dataframe
    :return: tuple of calculated intersection
    """
    geom_bounds = []
    pre_env = pre_df.to_crs(args.destination_crs).unary_union
    geom_bounds.append(pre_env)
    logger.debug(f'Pre bounds: {pre_env.bounds}')

    post_env = post_df.to_crs(args.destination_crs).unary_union
    geom_bounds.append(post_env)
    logger.debug(f'Post bounds: {post_env.bounds}')

    intersect = pre_env.intersection(post_env)
    assert intersect.area > 0, logger.critical('Pre and post imagery do not intersect')

    if aoi is not None:
        aoi_env = aoi.to_crs(args.destination_crs).unary_union
        geom_bounds.append(aoi_env)
        logger.debug(f'AOI bounds: {aoi_env.bounds}')
        intersect = intersect.intersection(aoi_env)
        assert intersect.area > 0, logger.critical('AOI does not intersect imagery')

    if in_poly_df is not None:
        in_poly_env = in_poly_df.to_crs(args.destination_crs).unary_union
        geom_bounds.append(in_poly_env)
        logger.debug(f'In poly bounds: {in_poly_env.bounds}')
        intersect = intersect.intersection(in_poly_env)
        assert intersect.area > 0, logger.critical('Building polygons do not intersect imagery/AOI')

    return intersect.bounds


def get_max_res(pre_df, post_df):
    """
    Calculates minimum resolution from two dataframes of raster footprints. Calculated on destination CRS units.
        Never mind the function name...I don't want to hear it.
    :param pre_df: geodataframe of raster footprints
    :param post_df: geodataframe of raster footprints
    :return: tuple of minimum resolution in (x, y)
    """
    res_list = list(pre_df.trans_res) + list(post_df.trans_res)
    x = max(x[0] for x in res_list)
    y = max(x[1] for x in res_list)
    return (x, y)


def bldg_poly_handler(poly_file):

    df = geopandas.read_file(poly_file)

    return df


def bldg_poly_process(df, intersect, dest_crs, out_file, out_shape, transform, ):

        def _clip_polys(input, mask):
            return geopandas.clip(input, mask)

        def _rasterize(in_feats, out_file, out_shape, transform, dst_crs):

            image = rasterio.features.rasterize(
                    in_feats.geometry,
                    out_shape=out_shape,
                    all_touched=True,
                    transform=transform,
                    )

            assert image.sum() > 0, logger.critical('Building polygon rasterization failed.')

            with rasterio.open(
                    out_file, 'w',
                    driver='GTiff',
                    dtype=rasterio.uint8,
                    count=1,
                    height=out_shape[0],
                    width=out_shape[1],
                    transform=transform,
                    crs=dst_crs) as dst:
                dst.write(image, indexes=1)

            return out_file

        poly_cords = [
                     (intersect[0], intersect[1]),
                     (intersect[2], intersect[1]),
                     (intersect[2], intersect[3]),
                     (intersect[0], intersect[3])
                     ]
        mask = geopandas.GeoDataFrame({'geometry': [Polygon(poly_cords)]}, crs=dest_crs)
        df = df.to_crs(dest_crs)

        assert mask.crs == df.crs, logger.critical('CRS mismatch')

        df = _clip_polys(df, mask)
        mosaic = _rasterize(df, out_file, out_shape, transform, dest_crs)

        return mosaic