from unittest import TestCase
import handler


class TestGetFiles(TestCase):

    def test_get_files(self):
        self.path = 'data/input/pre'
        self.result = handler.get_files(self.path)
        self.assertEqual(4, len(self.result))


class TestReprojectionHelper(TestCase):

    pass


class TestPostprocessAndWrite(TestCase):

    pass


class TestFilesClass(TestCase):

    pass

