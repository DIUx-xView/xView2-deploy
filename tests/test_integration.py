from tests.conftest import Args
from pathlib import Path
import pytest
import handler
import torch
import fiona
import rasterio.crs
from pytest import MonkeyPatch
from collections import namedtuple

# Todo: Return appropriate tensor for each image
# Todo: Class out our monkeypatches


@pytest.fixture(scope="class", autouse=True)
def monkeysession(request):
    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()


class MockLocModel:

    def __init__(self, *args, **kwargs):
        self.model_size = args[0]

    # Todo: this should return the correct tensor based on the image input. Currently returns the same tensor.
    # Todo: Ideally we store these with a CRS so we can reproject based on the mock args...some day
    # Mock inference results
    @staticmethod
    def forward(*args, **kwargs):
        arr = torch.load('tests/data/output/preds/preds_loc_0.pt')
        return arr


class MockClsModel:

    def __init__(self, *args, **kwargs):
        self.model_size = args[0]

    # Todo: this should return the correct tensor based on the image input. Currently returns the same tensor.
    # Mock inference results
    @staticmethod
    def forward(*args, **kwargs):
        arr = torch.load('tests/data/output/preds/preds_cls_0.pt')
        return arr


TestCase = namedtuple('TestCase', ['name',
                                   'pre_directory',
                                   'post_directory',
                                   'in_polys',
                                   'aoi_file',
                                   'pre_crs',
                                   'post_crs',
                                   'destination_crs',
                                   'destination_res',
                                   # Evaluation criteria
                                   'chip_num',
                                   'pre_checksum',
                                   'post_checksum',
                                   'damage_checksum',
                                   'overlay_checksum',
                                   'damage_polys',
                                   'expected_epsg'
                                   ])


""" # template for test cases

    TestCase(
        # Test name
        name,

        # Setup args
        pre_directory,
        post_directory,
        in_polys,
        aoi_file,
        pre_crs,
        post_crs,
        destination_crs,
        destination_res,

        # Evaluation criteria
        chip_num,
        pre_checksum,
        post_checksum,
        damage_checksum,
        overlay_checksum,
        damage_polys,
        expected_epsg
    )
"""


testcases = [
    TestCase(
        'integration_bldg_polys_no_aoi',
        'tests/data/input/pre',
        'tests/data/input/post',
        'tests/data/misc/bldg_polys.gpkg',
        None,
        None,
        None,
        None,
        None,
        # Evaluation criteria
        1,
        14928,
        26612,
        64780,
        50548,
        77,
        32615
        ),
    TestCase(
        'integration_no_polys_no_aoi',
        'tests/data/input/pre',
        'tests/data/input/post',
        None,
        None,
        None,
        None,
        None,
        None,
        # Evaluation criteria
        4,
        14928,
        26612,
        64780,
        50548,
        872,
        32615
        ),
    TestCase(
        'integration_no_poly_aoi',
        'tests/data/input/pre',
        'tests/data/input/post',
        None,
        'tests/data/misc/polygon_shapefile/intersect_polys.shp',
        None,
        None,
        None,
        None,
        # Evaluation criteria
        4,
        48631,
        15106,
        64780,
        33874,
        872,
        32615
    )
 ]


@pytest.fixture(scope='class', params=testcases, ids=[test.name for test in testcases])
def setup(output_path, request, monkeysession):
    # Pass args to handler
    monkeysession.setattr('argparse.ArgumentParser.parse_args', lambda x: Args(
        pre_directory=request.param.pre_directory,
        post_directory=request.param.post_directory,
        bldg_polys=request.param.in_polys,
        pre_crs=request.param.pre_crs,
        post_crs=request.param.post_crs,
        destination_crs=request.param.destination_crs,
        output_resolution=request.param.destination_res,
        aoi_file=request.param.aoi_file,
        output_directory=output_path
    ))

    # Mock CUDA devices
    monkeysession.setattr('torch.cuda.device_count', lambda: 2)
    monkeysession.setattr('torch.cuda.get_device_properties', lambda x: f'Mocked CUDA Device{x}')

    # Mock classes to mock inference
    monkeysession.setattr('handler.XViewFirstPlaceLocModel', MockLocModel)
    monkeysession.setattr('handler.XViewFirstPlaceClsModel', MockClsModel)

    # Call the handler
    handler.init()

    return request.param

class TestInput:

    @pytest.mark.parametrize('path,expected', [
        ('tests/data/input/pre', 4),
        ('tests/data/input/post', 6)
    ])
    def test_input(self, path, expected):
        """
        This is to ensure our inputs are correct. No code is tested here. If these fail expect that the input test
        data is incorrect and expect almost everything else to fail.
        :return:
        """
        assert len(list(Path(path).glob('**/*.tif'))) == expected


@pytest.mark.usefixtures('setup')
class TestIntegration:

    # Make sure files exist that we expect to exist
    @pytest.mark.parametrize('file', [
        ('mosaics/pre.tif'),
        ('mosaics/post.tif'),
        ('mosaics/damage.tif'),
        ('mosaics/overlay.tif'),
        ('vector/damage.gpkg'),
        ('vector/damage.geojson'),
        ('log/xv2.log')
    ])
    def test_is_files(self, setup, output_path, file):
        assert output_path.joinpath(file).is_file()

    # Make sure raster outputs look as we expect
    @pytest.mark.parametrize('file,expected', [
        ('mosaics/pre.tif', 'setup.pre_checksum'),
        ('mosaics/post.tif', 'setup.post_checksum'),
        ('mosaics/damage.tif', 'setup.damage_checksum'),
        ('mosaics/overlay.tif', 'setup.overlay_checksum')
    ])
    def test_pre_checksum(self, output_path, setup, file, expected):
        expected = eval(compile(expected, 'none', 'eval'))
        with rasterio.open(output_path.joinpath(file)) as src:
            assert src.checksum(1) == expected

    # Make sure directories contain the expected number of files
    @pytest.mark.parametrize('path,expected', [
        ('chips/pre', 'setup.chip_num'),
        ('chips/post', 'setup.chip_num'),
        ('loc', 'setup.chip_num'),
        ('dmg', 'setup.chip_num'),
        ('over', 'setup.chip_num'),
    ])
    def test_file_counts(self, setup, output_path, path, expected):
        expected = eval(compile(expected, 'none', 'eval'))
        assert len(list(output_path.joinpath(path).glob('**/*'))) == expected

    # Check out vector data contains the expected values
    @pytest.mark.parametrize('layer,expected', [
        ('damage', 'setup.damage_polys'),
        ('centroids', 'setup.damage_polys'),
        ('aoi', '1')
    ])
    def test_out_file_damage(self, setup, output_path, layer, expected):
        expected = eval(compile(expected, 'none', 'eval'))
        shapes = fiona.open(output_path.joinpath('vector/damage.gpkg'), layer=layer)
        assert len(shapes) == expected

    # Test that all vector layers have the correct CRS set
    @pytest.mark.parametrize('layer,epsg', [
        ('damage', {'init': 'epsg:32615'}),
        ('centroids', {'init': 'epsg:32615'}),
        ('aoi', {'init': 'epsg:32615'})
    ])
    def test_out_crs(self, setup, output_path, layer, epsg):
        with fiona.open(output_path.joinpath('vector/damage.gpkg'), layer=layer) as src:
            assert src.crs == epsg

    # Test for correct number of vector layers
    def test_out_file_layers(self, setup, output_path):
        assert len(fiona.listlayers(output_path.joinpath('vector/damage.gpkg'))) == 3

    # Test output rasters for correct CRS
    @pytest.mark.parametrize('file,expected', [
        ('mosaics/overlay.tif', 'setup.expected_epsg'),
        ('mosaics/pre.tif', 'setup.expected_epsg'),
        ('mosaics/post.tif', 'setup.expected_epsg')
    ])
    def test_out_epsg(self, setup, output_path, file, expected):
        expected = eval(compile(expected, 'none', 'eval'))
        with rasterio.open(output_path.joinpath(file)) as src:
            assert src.crs.to_epsg() == expected


class TestErrors:
    def test_geographic_crs(self):
        pass

    def test_errors(self):
        pass
