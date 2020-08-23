import unittest

import handler


class TestGetFiles(unittest.TestCase):

    def test_get_files(self):
        self.path = 'data/input/pre'
        self.result = handler.get_files(self.path)
        self.assertEqual(5, len(self.result))

    def test_get_files_mixed_case(self):
        self.path = 'data/input/post'
        self.result = handler.get_files(self.path)
        self.assertEqual(0, len(self.result))

    def test_get_files_no_recursive(self):
        self.path = 'data/input/post'
        self.result = handler.get_files(self.path, recursive=False)
        self.assertEqual(0, len(self.result))


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
        self.assertTrue(self.test_object.check_extent())