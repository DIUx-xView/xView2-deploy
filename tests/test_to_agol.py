from utils import to_agol
from utils import features
from pathlib import Path
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

class TestCreateDamagePolys:

    def test_damage_polys(self):
        file = Path('tests/data/output/mosaics/damage.tif')
        polys = features.create_polys([file], threshold=0)
        test = to_agol.create_damage_polys(polys)
        assert len(test) == 640

class TestAOIPolys:

    def test_aoi_polys(self):
        file = Path('tests/data/output/dmg/0_pre.tif')
        polys = features.create_polys([file])
        test = to_agol.create_aoi_poly(polys)
        assert test[0].geometry_type == 'Polygon'

class TestCentroids:

    def test_centroids(self):
        file = Path('tests/data/output/dmg/0_pre.tif')
        polys = features.create_polys([file], threshold=0)
        test = to_agol.create_centroids(polys)
        assert len(test) == 281