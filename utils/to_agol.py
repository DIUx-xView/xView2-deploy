import arcgis
from tqdm import tqdm
from loguru import logger

from utils.features import create_aoi_poly, create_centroids


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
            gis = agol_connect(user, password)
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
    gis = agol_connect(username=args.agol_user, password=args.agol_password)
    # Todo: this should be using the aoi and centroid shapes already created.
    dmg_df = polys.to_json()
    aoi = create_aoi_poly(polys)
    centroid_df = create_centroids(polys)

    result = agol_append(gis,
                         dmg_df,
                         args.agol_feature_service,
                         1)
    result = agol_append(gis,
                         aoi,
                         args.agol_feature_service,
                         2)
    result = agol_append(gis,
                         centroid_df,
                         args.agol_feature_service,
                         0)


def agol_connect(username, password):

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
        # Todo: This should use append IAW docs: https://developers.arcgis.com/python/api-reference/arcgis.features.toc.html?highlight=edit_features#arcgis.features.FeatureLayer.edit_features
        result = layer.edit_features(adds=batch, rollback_on_failure=True)

    logger.success(f'Appended {len(result.get("addResults"))} features to {layer.properties.name}')

    return True
