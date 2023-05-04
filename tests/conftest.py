from pathlib import Path

import geopandas
import pytest
import rasterio.crs
from pytest import MonkeyPatch
from shapely.geometry import Polygon

import handler
import utils.dataframe


# Todo: would like this to create a different output path for each parametrized test run
@pytest.fixture(scope="class")
def output_path(tmp_path_factory, request):
    return tmp_path_factory.mktemp(f"output_{request.module.testcases[0].name}")


#### Mock Argument class ####
class Args:
    def __init__(
        self,
        pre_directory="tests/data/input/pre",
        post_directory="tests/data/input/post",
        output_directory=None,
        n_procs=4,
        batch_size=1,
        num_workers=8,
        pre_crs=None,
        post_crs=None,
        destination_crs=None,
        dp_mode=True,
        output_resolution=None,
        save_intermediates=False,
        aoi_file="",
        bldg_polys=None,
    ):
        self.pre_directory = Path(pre_directory)
        self.post_directory = Path(post_directory)
        self.output_directory = output_directory
        self.n_procs = n_procs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pre_crs = pre_crs
        self.post_crs = post_crs
        self.destination_crs = destination_crs
        self.dp_mode = dp_mode
        self.output_resolution = output_resolution
        self.save_intermediates = save_intermediates
        self.aoi_file = aoi_file
        self.bldg_polys = bldg_polys


#### Dataframe fixtures ####


@pytest.fixture(scope="session")
def pre_df():
    args = Args(destination_crs=rasterio.crs.CRS.from_epsg(32615))
    files = handler.get_files(Path("tests/data/input/pre"))
    df = utils.dataframe.make_footprint_df(files)
    df = utils.dataframe.process_df(df, args.destination_crs)
    return df


@pytest.fixture(scope="session")
def post_df():
    args = Args(destination_crs=rasterio.crs.CRS.from_epsg(26915))
    files = handler.get_files("tests/data/input/post")
    df = utils.dataframe.make_footprint_df(files)
    df = utils.dataframe.process_df(df, args.destination_crs)
    return df


@pytest.fixture(scope="session")
def bldg_poly_df():
    df = geopandas.read_file("tests/data/misc/bldg_polys.gpkg")
    return df


@pytest.fixture(scope="session")
def no_intersect_df():
    coords = ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0))
    data = {"geometry": [Polygon(coords)]}
    df = geopandas.GeoDataFrame(data, geometry="geometry", crs=32612)
    return df


@pytest.fixture(scope="session")
def aoi_df():
    return geopandas.GeoDataFrame.from_file(
        "tests/data/misc/polygon_shapefile/intersect_polys.shp"
    )
