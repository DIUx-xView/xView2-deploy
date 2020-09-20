from unittest import TestCase
import os
import rasterio
from pathlib import Path

import raster_processing
import handler


class TestReproject(TestCase):

    def test_reproject(self):
        self.in_file = Path('data/input/pre/tile_337-10160.tif')
        self.dest_file = Path('data/output/resample.tif')
        self.result = raster_processing.reproject(self.in_file, self.dest_file, None)
        self.test = rasterio.open(self.result).crs
        self.assertEqual('EPSG:4326', self.test)


class TestCreateMosaic(TestCase):

    def setUp(self):
        self.files = handler.get_files(Path('data/input/pre'))
        self.out_file = Path('data/output/test_mosaic.tif')
        try:
            os.remove(self.out_file)
        except:
            pass

        self.result = raster_processing.create_mosaic(self.files, out_file=self.out_file)

    def test_output_exists(self):
        self.assertTrue(os.path.isfile(self.result))

    def test_correct_resolution(self):
        self.src = rasterio.open(self.result)
        self.assertEqual((0.6, 0.6), self.src.res)

    def test_correct_extent(self):
        self.src = rasterio.open(self.result)
        self.assertEqual((366642.60000000003, 4104511.1999999997), self.src.transform * (0, 0))


class TestGetIntersect(TestCase):

    def test_intersect_extent(self):
        self.test = raster_processing.get_intersect(
            Path('data/input/pre/tile_337-10160.tif'),
            Path('data/input/post/tile_31500-6161.tif')
        )
        self.assertEqual((364950.125, 4102656.6, 366951.60000000003, 4105049.875),
                         self.test)


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


class TestCheckDims(TestCase):

    pass
