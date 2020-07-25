import unittest

# import inference
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
        st1 = ['hurricane-matthew_00000351_pre_disaster.png']
        st2 = ['hurricane-matthew_00000351_pre_disaster.PNG']
        self.assertTrue(handler.file_valid_check(st1, st2))

    def test_string_unequal_length(self):
        st1 = ['hurricane-matthew_00000351_pre_disaster.PNG',
               'hurricane-matthew_00000351_post_disaster.png']
        st2 = ['hurricane-matthew_00000351_pre_disaster.PNG']
        self.assertFalse(handler.file_valid_check(st1, st2))

    def test_name_mismatch(self):
        pass

