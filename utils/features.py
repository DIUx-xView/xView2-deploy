import rasterio
from rasterio.features import shapes, dataset_features
from shapely.geometry import Polygon, shape
import geopandas


def create_polys(in_files, threshold=30):

    """
    Create palygons to use for feature creation.
    :param in_files: List of DMG files to create polygons from.
    :return: Shapely polygons.
    """

    polygons = []
    for idx, f in enumerate(in_files):
        src = rasterio.open(f)
        crs = src.crs
        transform = src.transform

        bnd = src.read(1)
        polygons += list(dataset_features(src, 1, geographic=False))

    # Create geo dataframe
    df = geopandas.GeoDataFrame.from_features(polygons, crs=crs)
    df.rename(columns={'val': 'dmg'}, inplace=True)

    # Drop damage of 0 (no building), dissolve by each damage level, and explode them back to single polygons
    df = df.dissolve(by='dmg').reset_index().drop(index=0)
    df = df.explode().reset_index(drop=True)

    # Apply our threshold
    df['area'] = df.geometry.area
    df = df[df.area >= threshold]

    # Fix geometry if not valid
    df.loc[~df.geometry.is_valid, 'geometry'] = df[~df.geometry.is_valid].geometry.apply(lambda x: x.buffer(0))

    return df.reset_index(drop=True)
