from utils import to_agol
import pytest
import arcgis
from .settings import *


class TestAGOLArgCheck:

    def test_no_user(self):
        with pytest.raises(ValueError):
            to_agol.agol_arg_check(None, 'test', 'test')

    def test_no_pass(self):
        with pytest.raises(ValueError):
            to_agol.agol_arg_check('test', None, 'test')

    def test_no_fs(self):
        with pytest.raises(ValueError):
            to_agol.agol_arg_check('test', 'test', None)

    def test_gis(self):
        assert to_agol.agol_arg_check(agol_user, agol_pass, agol_fs)

    def test_bad_creds(self):
        # arcgis package passes generic exception if connection is not made.
        with pytest.raises(Exception):
            assert to_agol.agol_arg_check(agol_user[:-1], agol_pass, agol_fs)

    def test_bad_layer(self):
        with pytest.raises(ValueError):
            to_agol.agol_arg_check(agol_user, agol_pass, agol_fs[:-1])
