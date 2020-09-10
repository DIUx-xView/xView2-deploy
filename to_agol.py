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
    aoi_polys = [geom for geom, val in features]
    hull = MultiPolygon(aoi_polys).convex_hull
    shape = arcgis.geometry.Geometry.from_shapely(hull)
    poly = arcgis.features.Feature(shape, attributes={'status': 'complete'})

    aoi_poly = [poly]

    return aoi_poly


def create_centroids(features):
    centroids = []
    for geom, val in features:
        esri_shape = arcgis.geometry.Geometry.from_shapely(geom.centroid)
        new_cent = arcgis.features.Feature(esri_shape, attributes={'dmg': val})
        centroids.append(new_cent)

    return centroids


def create_damage_polys(polys):
    polygons = []
    for geom, val in polys:
        esri_shape = arcgis.geometry.Geometry.from_shapely(geom)
        feature = arcgis.features.Feature(esri_shape, attributes={'dmg': val})
        polygons.append(feature)

    return polygons


def connect_gis(username, password):
    return arcgis.gis.GIS(username=username, password=password)


def agol_append(gis, src_feats, dest_fs, layer):

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
