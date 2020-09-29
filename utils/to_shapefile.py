import fiona
from shapely.geometry import mapping

def create_shapefile(polygons, out_shapefile, dest_crs):

    """
    Create shapefile from input polygons
    :param polygons: Polygons to export as shapefile.
    :param out_shapefile: Destination shapefile.
    :param dest_crs: Destination CRS.
    :return: None
    """

    shp_schema = {
            'geometry': 'Polygon',
            'properties': {'dmg': 'int'}
        }

    # Write out all the multipolygons to the same file
    with fiona.open(out_shapefile, 'w', 'ESRI Shapefile', shp_schema,
                    dest_crs) as shp:
        for polygon, px_val in polygons:
            shp.write({
                'geometry': mapping(polygon),
                'properties': {'dmg': int(px_val)}
            })

    return out_shapefile