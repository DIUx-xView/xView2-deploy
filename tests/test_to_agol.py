from utils import to_agol
import pytest
import arcgis
from unittest.mock import Mock


# Todo: parameterize this entire class
class TestAGOLArgCheck:

    def test_no_params(self):
        assert not to_agol.agol_arg_check(None, None, None)

    def test_no_user(self):
        assert not to_agol.agol_arg_check(None, 'test', 'test')

    def test_no_pass(self):
        assert not to_agol.agol_arg_check('test', None, 'test')

    def test_no_fs(self):
        assert not to_agol.agol_arg_check('test', 'test', None)

    def test_good_gis(self):
        # Mock a GIS object...this will also mock the gis.content.get method
        arcgis.gis.GIS = Mock()
        assert to_agol.agol_arg_check('test', 'test', 'test')

    def test_bad_creds(self):
        arcgis.gis.GIS = Mock(side_effect=Exception)
        with pytest.raises(Exception):
            assert to_agol.agol_arg_check('test', 'test', 'test')
            assert not to_agol.agol_arg_check('test', 'test', 'test')

    def test_bad_layer(self):
        arcgis.gis.GIS = Mock()
        gis = arcgis.gis.GIS()
        gis.content.get.return_value = None
        assert not to_agol.agol_arg_check('test', 'test', 'test')

    # Todo: Test connection timeout
    def test_timeout(self):
        pass
