import unittest

import handler


class TestGetFiles(unittest.TestCase):

    def test_get_files(self):
        self.path = 'data/input/pre'
        self.result = handler.get_files(self.path)
        self.assertEqual(4, len(self.result))


class TestReprojectionHelper(unittest.TestCase):

    pass


class TestPostprocessAndWrite(unittest.TestCase):

    pass


class TestFilesClass(unittest.TestCase):

    pass

