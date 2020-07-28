import unittest
import os

import inference
import handler


class TestGetFiles(unittest.TestCase):

    def test_get_files(self):
        self.path = 'input/post'
        self.result = handler.get_files(self.path)
        self.list_pre = ['hurricane-matthew_00000351_post_disaster.png']
        self.assertEqual(self.list_pre, self.result)

    def test_get_files_mixed_case(self):
        self.path = 'input/pre'
        self.result = handler.get_files(self.path)
        self.list_pre = ['hurricane-matthew_00000351_pre_disaster.PNG']
        self.assertEqual(self.list_pre, self.result)


class TestFileValidCheck(unittest.TestCase):

    def setUp(self):
        # TODO: Hold strings here to be passed to lists
        pass

    def test_string_equal_length(self):
        self.st1 = ['hurricane-matthew_00000351_pre_disaster.png']
        self.st2 = ['hurricane-matthew_00000351_pre_disaster.PNG']
        self.assertTrue(handler.string_len_check(self.st1, self.st2))

    def test_string_unequal_length(self):
        self.st1 = ['hurricane-matthew_00000351_pre_disaster.PNG',
                    'hurricane-matthew_00000351_post_disaster.png']
        self.st2 = ['hurricane-matthew_00000351_pre_disaster.PNG']
        self.assertFalse(handler.string_len_check(self.st1, self.st2))

    def test_no_files(self):
        pass

    def test_do_inference(self):
        pass


class TestFilesClass(unittest.TestCase):

    def setUp(self):
        self.test_object = handler.Files('hurricane-matthew_00000351_pre_disaster.PNG', 'hurricane-matthew_00000351_post_disaster.PNG')

    def test_infer(self):
        self.assertTrue(self.test_object)

    def test_base_num_match(self):
        self.assertTrue(self.test_object.check_base_num())


class TestInference(unittest.TestCase):

    def setUp(self):
        self.opts = inference.Options('sample_data/input/pre')

    def test_passed_path(self):
        self.assertEqual('sample_data/input/pre', self.opts.in_pre_path)
        self.assertFalse(self.opts.is_vis)

    def test_default_path(self):
        self.assertEqual('input/post', self.opts.in_post_path)
