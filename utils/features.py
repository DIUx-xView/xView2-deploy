import rasterio
from rasterio.features import dataset_features
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
    df.rename(columns={"val": "dmg"}, inplace=True)

    # Fix geometry if not valid
    df.loc[~df.geometry.is_valid, "geometry"] = df[
        ~df.geometry.is_valid
    ].geometry.apply(lambda x: x.buffer(0))

    # Drop damage of 0 (no building), dissolve by each damage level, and explode them back to single polygons
    df = df.dissolve(by="dmg").reset_index().drop(index=0)
    df = df.explode().reset_index(drop=True)

    # Apply our threshold
    df["area"] = df.geometry.area
    df = df[df.area >= threshold]

    return df.reset_index(drop=True)


def write_output(features, out_file, layer="features"):
    features.to_file(out_file, driver="GPKG", layer=layer)

    return out_file


def create_aoi_poly(features):

    """
    Create convex hull polygon encompassing damage polygons.
    :param features: Polygons to create hull around.
    :return: ARCGIS polygon.
    """
    hull = features.dissolve().convex_hull
    df = geopandas.GeoDataFrame.from_features(hull, crs=features.crs)
    return df


def create_centroids(features):

    """
    Create centroids from polygon features.
    :param features: Polygon features to create centroids from.
    :return: List of ARCGIS point features.
    """

    cent_df = geopandas.GeoDataFrame.from_features(features.centroid, crs=features.crs)
    cent_df["dmg"] = features.dmg
    return cent_df


def weight_dmg(features, destination_crs):

    poly = features.geometry.unary_union

    features.loc[features.dmg.isnull(), "dmg"] = 1
    features["dmg"] = round(
        ((features.geometry.area * features.dmg) / poly.area), ndigits=1
    )

    features = features.set_crs(destination_crs)
    features = features.dissolve(by="osmid", aggfunc=sum).reset_index(drop=True) # BUG: This will break if not using OSM data

    return features
