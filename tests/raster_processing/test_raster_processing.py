import unittest
import os

import raster_processing


class TestCreateMosaic(unittest.TestCase):

    def test_output_exists(self):
        # TODO: Remove this file prior to calling the function
        self.path = '../data/input/staging'
        self.files = [
            f'{os.path.abspath("input/post/hurricane-matthew_00000351_post_disaster.png")}',
            #f'{os.path.abspath("input/pre/hurricane-matthew_00000351_pre_disaster.PNG")}'
                      ]
        result = raster_processing.create_mosaic(self.path, self.files)
        self.assertTrue(os.path.isfile(result))
