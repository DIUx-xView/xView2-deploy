import unittest

import inference
import handler


class TestHandler(unittest.TestCase):

    def test_get_files(self):
        self.result = handler.get_files('input/post')
        self.list_pre = ['hurricane-matthew_00000351_post_disaster.png']
        self.assertEqual(self.list_pre, self.result)

    def test_get_files_mixed_case(self):
        self.result = handler.get_files('input/pre')
        self.list_pre = ['hurricane-matthew_00000351_pre_disaster.PNG']
        self.assertEqual(self.list_pre, self.result)

    def test_string_equal_length(self):
        self.st1 = ['hurricane-matthew_00000351_pre_disaster.png']
        self.st2 = ['hurricane-matthew_00000351_pre_disaster.PNG']
        self.assertTrue(handler.file_valid_check(self.st1, self.st2))

    def test_string_unequal_length(self):
        self.st1 = ['hurricane-matthew_00000351_pre_disaster.PNG',
               'hurricane-matthew_00000351_post_disaster.png']
        self.st2 = ['hurricane-matthew_00000351_pre_disaster.PNG']
        self.assertFalse(handler.file_valid_check(self.st1, self.st2))

    def test_name_mismatch(self):
        pass


class TestInference(unittest.TestCase):

    def setUp(self):
        self.opts = inference.Options('sample_data/input/pre')

    def test_passed_path(self):
        self.assertEqual('sample_data/input/pre', self.opts.in_pre_path)
        self.assertFalse(self.opts.is_vis)

    def test_default_path(self):
        self.assertEqual('input/post', self.opts.in_post_path)
