import rasterio
import pytest
from pathlib import Path

from utils import raster_processing
import handler


class TestReprojRes:

    def test_res(self):
        file = 'data/input/pre/tile_337-9136.tif'
        file2 = 'data/misc/no_crs/may24C350000e4102500n.jpg'
        test = raster_processing.get_reproj_res(file1, file2)
        assert test == (0.6, 0.6)


def test_get_intersect():
    test = raster_processing.get_intersect(
        Path('data/output/mosaics/pre.tif'),
        Path('data/output/mosaics/post.tif')
    )
    assert (-94.49960529516346, 37.06631597942802, -94.48623559881267, 37.07511383680346) == test


class TestReproject:

    def test_reproject_crs_set(self, tmp_path):
        # Test file with input having CRS set

        in_file = Path('data/input/pre/tile_337-10160.tif')
        dest_file = tmp_path / 'resample.tif'
        result = raster_processing.reproject(in_file, dest_file, None, 'EPSG:4326')
        with rasterio.open(result) as src:
            test = src.crs
        assert test == 'EPSG:4326'


    def test_reproject_no_crs_set(self, tmp_path):
        # Test file with input file having no CRS set

        in_file = Path('data/misc/no_crs/may24C350000e4102500n.jpg')
        dest_file = tmp_path / 'resample.tif'
        result = raster_processing.reproject(in_file, dest_file, 'EPSG:26915', 'EPSG:4326')
        with rasterio.open(result) as src:
            test = src.crs
        assert test == 'EPSG:4326'


    def test_reproject_no_crs(self, tmp_path):
        # Test file with no CRS set or passed

        in_file = Path('data/misc/no_crs/may24C350000e4102500n.jpg')
        dest_file = tmp_path / 'resample.tif'
        with pytest.raises(ValueError):
            result = raster_processing.reproject(in_file, dest_file, None, 'EPSG:4326')



class TestCheckDims:

    def test_check_dims_full_size(self):
        with rasterio.open(Path('data/output/chips/post/0_post.tif')) as src:
            arr = src.read()
        result = raster_processing.check_dims(arr, 1024, 1024)
        assert result.shape[1] == 1024
        assert result.shape[1] == 1024

    def test_check_dims_with_pad(self):
        with rasterio.open(Path('data/output/chips/post/0_post.tif')) as src:
            arr = src.read()
        result = raster_processing.check_dims(arr, 1500, 1500)
        assert result.shape[1] == 1500
        assert result.shape[1] == 1500

def test_create_mosaic(tmp_path):


    files = handler.get_files(Path('data/input/pre'))
    out_file = tmp_path / 'mosaic.tif'

    result = raster_processing.create_mosaic(files, out_file=out_file)

    # Test that we exported a file
    assert result.is_file()

    with rasterio.open(result) as src:
        # Test that the resolution is correct
        assert src.res == (0.6, 0.6)
        # Test that the extent is correct
        assert src.transform * (0, 0) == (366642.60000000003, 4104511.1999999997)


def test_create_chips(tmp_path):

    print(tmp_path)
    out_dir = tmp_path / 'chips'
    out_dir.mkdir()
    in_mosaic = Path('data/output/mosaics/pre.tif')
    intersect = (-94.49960529516346, 37.06631597942802, -94.48623559881267, 37.07511383680346)
    chips = raster_processing.create_chips(in_mosaic, out_dir, intersect)

    assert len(list(out_dir.iterdir())) == 6
    with rasterio.open(list(out_dir.iterdir())[0]) as src:
        assert src.height == 1024
        assert src.width == 1024
