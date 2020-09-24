import rasterio
from pathlib import Path

import raster_processing


def test_get_intersect():
    test = raster_processing.get_intersect(
        Path('data/output/mosaics/pre.tif'),
        Path('data/output/mosaics/post.tif')
    )
    assert (-94.49960529516346, 37.06631597942802, -94.48623559881267, 37.07511383680346) == test


def test_reproject_crs_set(tmp_path):
    # Test file with input having CRS set

    in_file = Path('data/input/pre/tile_337-10160.tif')
    dest_file = tmp_path / 'resample.tif'
    result = raster_processing.reproject(in_file, dest_file, None, 'EPSG:4326')
    with rasterio.open(result) as src:
        test = src.crs
    assert test == 'EPSG:4326'


def test_reproject_no_crs_set(tmp_path):
    # Test file with input file having no CRS set

    in_file = Path('data/misc/no_crs/may24C350000e4102500n.jpg')
    dest_file = tmp_path / 'resample.tif'
    result = raster_processing.reproject(in_file, dest_file, 'EPSG:26915', 'EPSG:4326')
    with rasterio.open(result) as src:
        test = src.crs
    assert test == 'EPSG:4326'
#
#
# class TestCreateMosaic(TestCase):
#
#     def setUp(self):
#         self.files = handler.get_files(Path('data/input/pre'))
#         self.out_file = Path('data/output/test_mosaic.tif')
#         try:
#             os.remove(self.out_file)
#         except:
#             pass
#
#         self.result = raster_processing.create_mosaic(self.files, out_file=self.out_file)
#
#     def test_output_exists(self):
#         self.assertTrue(os.path.isfile(self.result))
#
#     def test_correct_resolution(self):
#         self.src = rasterio.open(self.result)
#         self.assertEqual((0.6, 0.6), self.src.res)
#
#     def test_correct_extent(self):
#         self.src = rasterio.open(self.result)
#         self.assertEqual((366642.60000000003, 4104511.1999999997), self.src.transform * (0, 0))
#
#
# class TestCreateChips(unittest.TestCase):
#
#     def setUp(self):
#         self.out_dir = Path('data/output/chips/test_chips')
#         self.mosaic = TestCreateMosaic()
#         self.in_mosaic = self.mosaic.result
#
#     def test_chip_exist(self):
#         self.assertEqual(0, len())
#
#     def test_get_intersect_win(self):
#         self.intersect = (-94.62862733666518, 36.997318285256874, -94.55892307170559, 37.06518001454033)
#         self.mosaic = Path('/Users/lb/Documents/PycharmProjects/xView2_FDNY/tests/data/output/test_mosaic.tif')
#         self.test = 'Window(col_off=211, row_off=10114, width=10853, height=11148)'
#         self.result = raster_processing.get_intersect_win(self.mosaic, self.intersect)
#         self.assertEqual(self.test, self.result)
#
#
# class TestCheckDims(TestCase):
#
#     pass
