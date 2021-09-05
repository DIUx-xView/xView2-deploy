from tests.conftest import Args
from pathlib import Path
import pytest
import handler
import torch
import fiona
import rasterio.crs
from pytest import MonkeyPatch

# Todo: Return appropriate tensor for each image
# Todo: Class out our monkeypatches

@pytest.fixture(scope="class", autouse=True)
def monkeypatch_for_class(request):
    """
    Allows for the use of MonkeyPatch inside our test classes
    :param request:
    :return:
    """
    request.cls.monkeypatch = MonkeyPatch()


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


def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    argnames = sorted(funcarglist[0])
    metafunc.parametrize(
        argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
    )


class TestInput:

    params = {
        'test_input': [dict(file='tests/data/input/pre', expected=4),
                       dict(file='tests/data/input/post', expected=6)]
    }

    def test_input(self, file, expected):
        """
        This is to ensure our inputs are correct. No code is tested here. If these fail expect that the input test
        data is incorrect and expect almost everything else to fail.
        :return:
        """
        assert len(list(Path(file).glob('**/*'))) == expected


@pytest.mark.parametrize('aoi', [
                             pytest.param('tests/data/misc/polygon_shapefile/intersect_polys.shp'),
                             pytest.param(None)],
                         scope='class')
class TestIntegration:


    params = {
    'test_is_files': [dict(file='mosaics/pre.tif'),
                       dict(file='mosaics/post.tif'),
                       dict(file='mosaics/damage.tif'),
                       dict(file='mosaics/overlay.tif'),
                       dict(file='vector/damage.gpkg'),
                       dict(file='log/xv2.log')],
    'test_img_checksum': [dict(file='mosaics/pre.tif', expected=0),
                          dict(file='mosaics/post.tif', expected=0),
                          dict(file='mosaics/damage.tif', expected=0),
                          dict(file='mosaics/overlay.tif', expected=0)],
    'test_file_counts': [dict(folder='chips/pre', expected=4),
                         dict(folder='chips/post', expected=4),
                         dict(folder='loc', expected=4),
                         dict(folder='dmg', expected=4),
                         dict(folder='over', expected=4)],
    'test_out_file_damage': [dict(layer='damage', expected=872),
                             dict(layer='centroids', expected=872),
                             dict(layer='aoi', expected=1)],
    'test_out_crs': [dict(layer='damage'),
                     dict(layer='centroids'),
                     dict(layer='aoi')],
    'test_out_file_layers': [dict(file='vector/damage.gpkg', expected=3)],
    'test_out_epsg': [dict(file='mosaics/overlay.tif', expected=32615),
                      dict(file='mosaics/pre.tif', expected=32615),
                      dict(file='mosaics/post.tif', expected=32615)]
    }

    @pytest.fixture(autouse=True, scope='class')
    def setup(self, output_path, aoi):
        # Pass args to handler
        self.monkeypatch.setattr('argparse.ArgumentParser.parse_args', lambda x: Args(
            output_path=output_path,
            aoi_file=aoi
        )
                                 ),

        # Mock CUDA devices
        self.monkeypatch.setattr('torch.cuda.device_count', lambda: 2)
        self.monkeypatch.setattr('torch.cuda.get_device_properties', lambda x: f'Mocked CUDA Device{x}')

        # Mock classes to mock inference
        self.monkeypatch.setattr('handler.XViewFirstPlaceLocModel', MockLocModel)
        self.monkeypatch.setattr('handler.XViewFirstPlaceClsModel', MockClsModel)

        # Call the handler
        handler.init()

    # Make sure files exist that we expect to exist
    def test_is_files(self, output_path, file):
        assert output_path.joinpath(file).is_file()

    # Make sure raster outputs look as we expect
    def test_img_checksum(self, output_path, file, expected):
        with rasterio.open(output_path.joinpath(file)) as src:
            assert src.checksum() == expected

    # Make sure directories contain the expected number of files
    def test_file_counts(self, output_path, folder, expected):
        assert len(list(output_path.joinpath(folder).glob('**/*'))) == expected

    # Check out vector data contains the expected values
    def test_out_file_damage(self, output_path, layer, expected):
        shapes = fiona.open(output_path.joinpath('vector/damage.gpkg'), layer=layer)
        assert len(shapes) == expected

    # Test that all vector layers have the correct CRS set
    def test_out_crs(self, output_path, layer):
        with fiona.open(output_path.joinpath('vector/damage.gpkg'), layer=layer) as src:
            assert src.crs == {'init': 'epsg:32615'}

    # Test for correct number of vector layers
    def test_out_file_layers(self, output_path, file, expected):
        assert len(fiona.listlayers(output_path.joinpath(file))) == expected

    # Test output rasters for correct CRS
    def test_out_epsg(self, output_path, file, expected):
        with rasterio.open(output_path.joinpath(file)) as src:
            assert src.crs.to_epsg() == expected


class TestNoCUDA:

    @pytest.fixture(scope='class', autouse=True)
    def setup(self, output_path):
        # Pass args to handler
        self.monkeypatch.setattr('argparse.ArgumentParser.parse_args', lambda x: Args(
            output_path=output_path
        )
                                 ),

        # Mock CUDA devices
        self.monkeypatch.setattr('torch.cuda.device_count', lambda: 0)
        self.monkeypatch.setattr('torch.cuda.get_device_properties', lambda x: f'Mocked CUDA Device{x}')

        # Mock classes to mock inference
        self.monkeypatch.setattr('handler.XViewFirstPlaceLocModel', MockLocModel)
        self.monkeypatch.setattr('handler.XViewFirstPlaceClsModel', MockClsModel)

    def test_pass(self):
        # Call the handler
        with pytest.raises(ValueError):
            handler.init()

    def test_log(self, output_path):
        assert (output_path / 'log' / 'xv2.log').is_file()


@pytest.mark.skip
class TestExperiment:
    """
    Experiment class to run the handler and view the output path for A/B testing etc.
    Skip mark should be set when not in use.
    """

    @pytest.fixture(scope='class', autouse=True)
    def setup(self, output_path):
        # Pass args to handler
        self.monkeypatch.setattr('argparse.ArgumentParser.parse_args', lambda x: Args(
            output_path=output_path
        )
                                 ),

        # Mock CUDA devices
        self.monkeypatch.setattr('torch.cuda.device_count', lambda: 2)
        self.monkeypatch.setattr('torch.cuda.get_device_properties', lambda x: f'Mocked CUDA Device{x}')

        # Mock classes to mock inference
        self.monkeypatch.setattr('handler.XViewFirstPlaceLocModel', MockLocModel)
        self.monkeypatch.setattr('handler.XViewFirstPlaceClsModel', MockClsModel)

        # Call the handler
        handler.init()

    def test_experiment(self, output_path):
        # Fail the test so we can view the output path
        assert output_path == 0
