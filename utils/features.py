import rasterio
from rasterio.features import shapes
from shapely.geometry import Polygon, shape


def make_valid(ob):
    # This is a hack until shapely is updated with shapely.validation.make_valid
    if ob.is_valid:
        return ob
    else:
        return ob.buffer(0)



def create_polys(in_files):

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
        polygons += list(shapes(bnd, transform=transform))

    features = [(make_valid(Polygon(shape(geom))), val) for geom, val in polygons if val > 0]

    return [feature for feature in features if feature[0].area > threshold]
