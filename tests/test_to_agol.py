from utils import to_agol
import pytest
import arcgis
from unittest.mock import Mock


class TestAGOLArgCheck:

    @pytest.mark.parametrize('user,password,fs,expected', [
        pytest.param(None, None, None, False, id='no_params'),
        pytest.param(None, 'test', 'test', False, id='no_user'),
        pytest.param('test', None, 'test', False, id='no_pass'),
        pytest.param('test', 'test', None, False, id='no_fs'),
        pytest.param('test', 'test', 'test', True, id='good_to_agol')
    ])
    def test_to_agol(self, user, password, fs, expected):
        # Mock a GIS object...this will also mock the gis.content.get method
        arcgis.gis.GIS = Mock()
        assert to_agol.agol_arg_check(user, password, fs) == expected

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
