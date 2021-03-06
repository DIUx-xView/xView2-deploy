import arcgis
from tqdm import tqdm
from shapely.geometry import MultiPolygon
from loguru import logger


# Enable .from_shapely for building AGOL features from shapely features.
@classmethod
def from_shapely(cls, shapely_geometry):
    return cls(shapely_geometry.__geo_interface__)


arcgis.geometry.BaseGeometry.from_shapely = from_shapely


def agol_arg_check(user, password, fs_id):

    """
    Checks that AGOL parameters are present for proper operation.
    :param args: Arguments
    :return: True if arguments are present to accomplish AGOL push. False if not.
    """

    # Check that all parameters have been passed to args.
    if any((user, password, fs_id)):
        # Return false if all arguments were not passed
        if not all((user, password, fs_id)):
            logger.warning('Missing required AGOL parameters. Skipping AGOL push.')
            return False

        # Test the AGOL connection
        try:
            gis = connect_gis(user, password)
        # Todo: Also need to catch instance of nothing returned (ie. no internet connection)
        except Exception as ex:  # Incorrect user/pass raises an exception
            # Todo: this message is not entirely accurate. Check for connection
            logger.warning(f'Unable to connect to AGOL. Check username and password. Skipping AGOL push {ex}')
            return False

        # Test that we can get the passed layer
        layer = gis.content.get(fs_id)
        if layer:
            return True
        else:
            logger.warning(f'AGOL layer \'{fs_id}\' not found.')
            return False

    # Return false if no arguments were passed
    else:
        logger.warning('Attempt to connect to AGOL failed. Check the arguments and try again.')
        return False


def agol_helper(args, polys):
    gis = connect_gis(username=args.agol_user, password=args.agol_password)

    dmg_polys = create_damage_polys(polys)
    aoi_poly = create_aoi_poly(polys)  # TODO: Should this be included in the shapefile?
    centroids = create_centroids(polys)

    result = agol_append(gis,
                         dmg_polys,
                         args.agol_feature_service,
                         1)
    result = agol_append(gis,
                         aoi_poly,
                         args.agol_feature_service,
                         2)
    result = agol_append(gis,
                         centroids,
                         args.agol_feature_service,
                         0)


def create_aoi_poly(features):

    """
    Create convex hull polygon encompassing damage polygons.
    :param features: Polygons to create hull around.
    :return: ARCGIS polygon.
    """
    # Todo: This should be a rectangle of the intersect.
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


    logger.info('Attempting to append features to ArcGIS')
    layer = gis.content.get(dest_fs).layers[int(layer)]
    for batch in tqdm(batch_gen(src_feats, 1000)):
        result = layer.edit_features(adds=batch, rollback_on_failure=True)

    logger.success(f'Appended {len(result.get("addResults"))} features to {layer.properties.name}')

    return True
