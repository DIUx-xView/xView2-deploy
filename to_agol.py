import arcgis
import rasterio
import shapely.geometry
from tqdm import tqdm
from rasterio.features import shapes
from shapely.geometry import Polygon, shape, MultiPolygon



# Enable .from_shapely for building AGOL features from shapely features.
@classmethod
def from_shapely(cls, shapely_geometry):
    return cls(shapely_geometry.__geo_interface__)

arcgis.geometry.BaseGeometry.from_shapely = from_shapely


def agol_arg_check(args):

    """
    Checks that AGOL parameters are present for proper operation.
    :param args: Arguments
    :return: True if arguments are present to accomplish AGOL push. False if not.
    """

    agol_args = [args.agol_user,
                 args.agol_password,
                 args.agol_feature_service
                 ]

    if any([agol_args]):
        if not args.agol_user:
            print('Missing AGOL username. Skipping AGOL push.')
            return False
        elif not args.agol_password:
            print('Missing AGOL password. Skipping AGOL push.')
            return False
        elif not args.agol_feature_service:
            print('Missing AGOL damage feature service ID. Skipping AGOL push.')
            return False
    else:
        return False

    # If everything is in place...
    return True


def create_polys(in_files):
    # TODO: Combine this with raster_processing.create_shapefile polygon creation.
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


def create_aoi_poly(features):

    """
    Create convex hull polygon encompassing damage polygons.
    :param features: Polygons to create hull around.
    :return: ARCGIS polygon.
    """

    aoi_polys = [geom for geom, val in features]
    hull = MultiPolygon(aoi_polys).convex_hull
    shape = arcgis.geometry.Geometry.from_shapely(hull)
    poly = arcgis.features.Feature(shape, attributes={'status': 'complete'})

    aoi_poly = [poly]

    return aoi_poly


def create_centroids(features):

    """
    Create centroids from polygon features.
    :param features: Polygon features to create centroids from.
    :return: List of ARCGIS point features.
    """

    centroids = []
    for geom, val in features:
        esri_shape = arcgis.geometry.Geometry.from_shapely(geom.centroid)
        new_cent = arcgis.features.Feature(esri_shape, attributes={'dmg': val})
        centroids.append(new_cent)

    return centroids


def create_damage_polys(polys):

    """
    Create ARCGIS polygon features.
    :param polys: Polygons to create ARCGIS features from.
    :return: List of ARCGIS polygon features.
    """

    polygons = []
    for geom, val in polys:
        esri_shape = arcgis.geometry.Geometry.from_shapely(geom)
        feature = arcgis.features.Feature(esri_shape, attributes={'dmg': val})
        polygons.append(feature)

    return polygons


def connect_gis(username, password):

    """
    Create a ArcGIS connection
    :param username: AGOL username.
    :param password: AGOL password.
    :return: AGOL GIS object.
    """

    return arcgis.gis.GIS(username=username, password=password)


def agol_append(gis, src_feats, dest_fs, layer):

    """
    Add features to AGOL feature service.
    :param gis: AGOL connection.
    :param src_feats: Features to add.
    :param dest_fs: Destination feature service ID.
    :param layer: Layer number to append.
    :return: True if successful.
    """

    def batch_gen(iterable, n=1):
        l = len(iterable)
        for idx in range(0, l, n):
            yield iterable[idx:min(idx + n, l)]


    print('Attempting to append features to ArcGIS')
    layer = gis.content.get(dest_fs).layers[int(layer)]
    for batch in tqdm(batch_gen(src_feats, 1000)):
        result = layer.edit_features(adds=batch, rollback_on_failure=True)

    #print(f'Appended {len(result.get("addResults"))} features to {layer.properties.name}')

    return True
