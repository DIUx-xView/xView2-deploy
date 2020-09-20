import rasterio
from rasterio.features import shapes
from shapely.geometry import Polygon, shape

def create_polys(in_files):

    """
    Create palygons to use for feature creation.
    :param in_files: DMG files to create polygons from.
    :return: Polygons from dmg files.
    """

    polygons = []
    for idx, f in enumerate(in_files):
        src = rasterio.open(f)
        crs = src.crs
        transform = src.transform

        bnd = src.read(1)
        polys = list(shapes(bnd, transform=transform))

        for geom, val in polys:
            if val == 0:
                continue
            polygons.append((Polygon(shape(geom)), val))

    return polygons