from utils import to_shapefile
from unittest import TestCase
from utils.features import create_polys
from handler import get_files
from pathlib import Path
import rasterio.crs


def test_create_shapefile(tmp_path):

    files = get_files(Path('tests/data/output/dmg'))
    polys = create_polys(files)
    out_path = tmp_path / 'shapes'
    out_path.mkdir()
    shapefile = to_shapefile.create_shapefile(polys, out_path / 'shapes.shp')
    assert Path.is_file(shapefile)
