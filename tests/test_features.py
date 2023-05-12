from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon

import handler
import utils.features
from utils import features


class TestCreatePolys:
    def test_damage_poly_no_threshold(self):
        file = Path("tests/data/output/mosaics/damage.tif")
        polys = features.create_polys([file], threshold=0)
        assert len(polys) == 640

    def test_damage_poly_threshold(self):
        file = Path("tests/data/output/mosaics/damage.tif")
        polys = features.create_polys([file])
        assert len(polys) == 409

    def test_damage_poly_multi_files(self, tmp_path):
        # Test for bug #39
        file1 = Path("tests/data/output/dmg/0_pre.tif")
        file2 = Path("tests/data/output/dmg/1_pre.tif")
        polys = features.create_polys([file1, file2], threshold=0)
        assert len(polys) == 353

    def test_combine_poly_with_thresh(self):
        file1 = Path("tests/data/output/dmg/0_pre.tif")
        file2 = Path("tests/data/output/dmg/1_pre.tif")
        polys = features.create_polys([file1, file2])
        assert len(polys) == 252

    def test_geom_valid(self):
        # Test for bug #43
        file1 = Path("tests/data/output/dmg/0_pre.tif")
        file2 = Path("tests/data/output/dmg/1_pre.tif")
        polys = features.create_polys([file1, file2], threshold=0)
        assert all(polys.geometry.is_valid)


class TestCreateOutput:
    def test_create_output(self, tmp_path):
        files = handler.get_files(Path("tests/data/output/dmg"))
        polys = features.create_polys(files)
        out_path = tmp_path / "vector"
        out_path.mkdir()
        shapefile = features.write_output(polys, out_path / "damage.geojson")
        assert Path.is_file(shapefile)


class TestAOIPolys:
    def test_aoi_polys(self):
        file = Path("tests/data/output/dmg/0_pre.tif")
        polys = features.create_polys([file])
        test = utils.features.create_aoi_poly(polys)
        assert test.geometry[0].geom_type == "Polygon"
        assert len(test) == 1


class TestCentroids:
    def test_centroids(self):
        file = Path("tests/data/output/mosaics/damage.tif")
        polys = features.create_polys([file], threshold=0)
        test = utils.features.create_centroids(polys)
        assert len(test) == 640


class TestWeightDMG:
    def test_weight_dmg(self):
        data = {
            "dmg": [1, 1, 2, 3],
            "index": [1, 1, 1, 1],
            "geometry": [
                Polygon(((0, 0), (0, 1), (1, 1), (1, 0))),
                Polygon(((0, 1), (0, 2), (1, 2), (1, 1))),
                Polygon(((1, 0), (1, 1), (2, 1), (2, 0))),
                Polygon(((1, 1), (1, 2), (2, 2), (2, 1))),
            ],
        }
        df = gpd.GeoDataFrame(data)
        test = features.weight_dmg(df)
        assert test.dmg.values[0] == 1.75
