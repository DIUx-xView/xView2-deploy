from pathlib import Path
from utils import features
import pytest


class TestCreatePolys:

    def test_damage_poly_no_threshold(self):
        file = Path('tests/data/output/mosaics/damage.tif')
        polys = features.create_polys([file], threshold=0)
        assert len(polys) == 640

    def test_damage_poly_threshold(self):
        file = Path('tests/data/output/mosaics/damage.tif')
        polys = features.create_polys([file])
        assert len(polys) == 409

    def test_damage_poly_multi_files(self):
        # Test for bug #39
        file1 = Path('tests/data/output/dmg/0_pre.tif')
        file2 = Path('tests/data/output/dmg/1_pre.tif')
        polys = features.create_polys([file1, file2], threshold=0)
        assert len(polys) == 353

    def test_geom_valid(self):
        # Test for bug #43
        file = Path('tests/data/output/mosaics/damage.tif')
        polys = features.create_polys([file], threshold=0)
        shapes = [x[0].is_valid for x in polys]
        assert all(shapes)
