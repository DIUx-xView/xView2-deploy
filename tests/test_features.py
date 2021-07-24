from pathlib import Path
from utils import features
import pytest


class TestCreatePolys:

    def test_damage_poly_no_threshold(self):
        file = Path('tests/data/output/mosaics/damage.tif')
        polys = features.create_polys([file], threshold=0)
        assert len(polys) == 628

    def test_damage_poly_threshold(self):
        file = Path('tests/data/output/mosaics/damage.tif')
        polys = features.create_polys([file], threshold=50) # Todo: Pretty large discrpancy between what's reported by shapely vs. QGIS
        assert len(polys) == 421

    def test_damage_poly_multi_files(self):
        # Test for bug #39
        file1 = Path('tests/data/output/dmg/0_pre.tif')
        file2 = Path('tests/data/output/dmg/1_pre.tif')
        polys = features.create_polys([file1, file2])
        assert len(polys) == 326

    def test_geom_valid(self):
        # Test for bug #43
        file = Path('tests/data/output/mosaics/damage.tif')
        polys = features.create_polys([file])
        shapes = [x[0].is_valid for x in polys]
        assert all(shapes)
