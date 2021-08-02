import fiona
import fiona.crs
from shapely.geometry import mapping


def create_shapefile(polygons, out_shapefile):
    polygons.to_file(out_shapefile)

    return out_shapefile