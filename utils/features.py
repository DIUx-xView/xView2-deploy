from pathlib import Path
from typing import Union

import geopandas
import rasterio
from rasterio.features import dataset_features


def create_polys(in_files: list, threshold: int = 30) -> geopandas.GeoDataFrame:
    """Create polygons for for vector feature creation

    Args:
        in_files (list): List of input files to parse
        threshold (int, optional): Threshold for filtering small features. Features with an area small than 'threshold' will be dropped. Area units are in CRS linear unit. Defaults to 30.

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame of features meeting 'threshold'.
    """

    polygons = []
    for idx, f in enumerate(in_files):
        src = rasterio.open(f)
        crs = src.crs
        polygons += list(dataset_features(src, 1, geographic=False))

    df = geopandas.GeoDataFrame.from_features(polygons, crs=crs)
    df.rename(columns={"val": "dmg"}, inplace=True)

    # Fix geometry if not valid
    df.loc[~df.geometry.is_valid, "geometry"] = df.geometry.apply(lambda x: x.buffer(0))

    # Drop damage of 0 (no building)
    df = df.drop(df[df.dmg == 0].index)

    # Apply our threshold
    df["area"] = df.geometry.area
    df = df[df.area >= threshold]

    return df.reset_index(drop=True)


def write_output(
    features: geopandas.GeoDataFrame,
    out_file: Union[Path, str],
    layer: str = "features",
) -> Union[Path, str]:
    """Write GeoDataFrame to file.

    Args:
        features (geopandas.GeoDataFrame): Input features
        out_file (Union[Path, str]): Path or string of output file
        layer (str, optional): Layer in file to write features to. Defaults to "features".

    Returns:
        Union[Path, str]: Returns Path or string (as passed to func) of output file.
    """
    features.to_file(out_file, driver="GPKG", layer=layer)

    return out_file


def create_aoi_poly(polygons: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
    """Create GeoDataFrame AOI (convex hull) from input GeoDataFrame.

    Args:
        polygons (geopandas.GeoDataFrame): GeoDataFrame of input features.

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame of AOI feature in same CRS as input features.
    """
    hull = polygons.unary_union.convex_hull
    df = geopandas.GeoDataFrame({"geometry": [hull]}, crs=polygons.crs)
    return df


def create_centroids(features):
    """Create GeoDataFrame of Centroids for input polygons in GeoDataFrame.

    Args:
        features (geopandas.GeoDataFrame): Input GeoDataFrame of input polygons.

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame of centroids for input polygons.
    """

    cent_df = geopandas.GeoDataFrame.from_features(features.centroid, crs=features.crs)
    cent_df["dmg"] = features.dmg
    return cent_df


def weight_dmg(features: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
    """Given a GeoDataFrame of input features, calculate the weighted damage by summing the weights of all damage.

    Args:
        features (geopandas.GeoDataFrame): Input features.

    Returns:
        geopandas.GeoDataFrame: Weighted damage score for input features.
    """
    poly = features.geometry.unary_union

    features.loc[features.dmg.isnull(), "dmg"] = 1
    features["dmg"] = round(
        ((features.geometry.area * features.dmg) / poly.area), ndigits=2
    )

    features = features.dissolve(by="index", aggfunc=sum)

    return features
