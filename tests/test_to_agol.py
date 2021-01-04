from utils import to_agol
from utils import features
from pathlib import Path
import pytest
import arcgis
from .settings import *


class TestAGOLArgCheck:

    def test_no_params(self):
        assert not to_agol.agol_arg_check(None, None, None)

    def test_no_user(self):
        assert not to_agol.agol_arg_check(None, 'test', 'test')

    def test_no_pass(self):
        assert not to_agol.agol_arg_check('test', None, 'test')

    def test_no_fs(self):
        assert not to_agol.agol_arg_check('test', 'test', None)

    def test_gis(self):
        assert to_agol.agol_arg_check(agol_user, agol_pass, agol_fs)

    def test_bad_creds(self):
        # arcgis package passes generic exception if connection is not made.
        with pytest.raises(Exception):
            assert to_agol.agol_arg_check(agol_user[:-1], agol_pass, agol_fs)

    def test_bad_layer(self):
        assert not to_agol.agol_arg_check(agol_user, agol_pass, agol_fs[:-1])


class TestCreateDamagePolys:

    def test_damage_polys(self):
        file = Path('data/output/dmg/0_pre.tif')
        polys = features.create_polys([file])
        test = to_agol.create_damage_polys(polys)
        assert len(test) == 264
