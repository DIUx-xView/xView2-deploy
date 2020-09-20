from utils import to_shapefile
from unittest import TestCase
from utils.features import create_polys
from handler import get_files
from pathlib import Path


class Test(TestCase):

    def test_create_shapefile(self):
        self.files = get_files(Path('data/output/dmg'))
        self.polys = create_polys(self.files)
        self.shapefile = to_shapefile(self.polys, Path('~/Downloads'), 'EPSG:4326')
