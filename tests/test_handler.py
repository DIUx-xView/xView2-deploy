import pytest

import handler


class TestGetFiles:
    def test_get_files(self):
        path = "tests/data/input/pre"
        result = handler.get_files(path)
        assert len(result) == 4

    def test_no_files(self):
        with pytest.raises(AssertionError):
            handler.get_files("tests/data/empty_test_dir")


def test_reprojection_helper():
    pass


def test_post_process_and_write():
    pass


def test_files_class():
    pass
