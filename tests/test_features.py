from pathlib import Path
from utils import features


class TestCreatePolys:

    def test_damage_poly(self):
        file = Path('tests/data/output/dmg/0_pre.tif')
        polys = features.create_polys([file])
        assert len(polys) == 264

    def test_damage_poly_mult_files(self):
        # Test for bug #39
        file1 = Path('tests/data/output/dmg/0_pre.tif')
        file2 = Path('tests/data/output/dmg/1_pre.tif')
        polys = features.create_polys([file1, file2])
        assert len(polys) == 326

    def test_geom_valid(self):
        # Test for bug #43
        file1 = Path('tests/data/output/dmg/0_pre.tif')
        file2 = Path('tests/data/output/dmg/1_pre.tif')
        polys = features.create_polys([file1, file2])
        shapes = [x[0].is_valid for x in polys]
        assert all(shapes)
