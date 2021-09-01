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



@pytest.fixture(scope='class', autouse=True)
def output_path(tmp_path_factory):
    return tmp_path_factory.mktemp('output')


@pytest.fixture(scope='class', autouse=True)
def staging_path(tmp_path_factory):
    return tmp_path_factory.mktemp('staging')


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


@pytest.fixture(scope="class", autouse=True)
def monkeypatch_for_class(request):
    """
    Allows for the use of MonkeyPatch inside our test classes
    :param request:
    :return:
    """
    request.cls.monkeypatch = MonkeyPatch()


class TestInput:

    def test_input(self):
        """
        This is to ensure our inputs are correct. No code is tested here. If these fail expect that the input test
        data is incorrect and expect almost everything else to fail.
        :return:
        """
        assert len(list(Path('tests/data/input/pre').glob('**/*'))) == 4
        assert len(list(Path('tests/data/input/post').glob('**/*'))) == 6


class TestGood:

    @pytest.fixture(scope='class', autouse=True)
    def setup(self, staging_path, output_path):
        # Pass args to handler
        self.monkeypatch.setattr('argparse.ArgumentParser.parse_args', lambda x: Args(
            staging_path=staging_path,
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

    # Make sure files exist that we expect to exist
    @pytest.mark.parametrize('file', [
        pytest.param('mosaics/pre.tif', id='pre_mosaic_is_file'),
        pytest.param('mosaics/post.tif', id='post_mosaic_is_file'),
        pytest.param('mosaics/damage.tif', id='damage_mosaic_is_file'),
        pytest.param('mosaics/overlay.tif', id='overlay_mosaic_is_file'),
        pytest.param('vector/damage.gpkg', id='damage_vector_is_file'),
        pytest.param('log/xv2.log', id='log_file_is_file')
    ])
    def test_is_files(self, output_path, file):
        assert output_path.joinpath(file).is_file()

    # Make sure raster outputs look as we expect
    @pytest.mark.parametrize('file,expected', [
        pytest.param('mosaics/pre.tif', 0, id='pre_mosaic_checksum'),
        pytest.param('mosaics/post.tif', 0, id='post_mosaic_checksum'),
        pytest.param('mosaics/damage.tif', 0, id='damage_mosaic_checksum'),
        pytest.param('mosaics/overlay.tif', 0, id='overlay_mosaic_checksum'),
    ])
    def img_checksum(self, output_path, file, expected):
        with rasterio.open(output_path.joinpath(file)) as src:
            assert src.checksum() == expected

    # Make sure directories contain the expected number of files
    @pytest.mark.parametrize('folder,expected', [
        pytest.param('chips/pre', 6, id='count_pre_chips'),
        pytest.param('chips/post', 6, id='count_post_chips'),
        pytest.param('loc', 6, id='count_loc_chips'),
        pytest.param('dmg', 6, id='count_damage'),
        pytest.param('over', 6, id='count_overlay_chips')
    ])
    def test_file_counts(self, output_path, folder, expected):
        assert len(list(output_path.joinpath(folder).glob('**/*'))) == expected

    # Check out vector data contains the expected values
    @pytest.mark.parametrize('layer, expected', [('damage', 1272), ('centroids', 1272), ('aoi', 1)])
    def test_out_file_damage(self, output_path, layer, expected):
        shapes = fiona.open(output_path.joinpath('vector/damage.gpkg'), layer=layer)
        assert len(shapes) == expected

    # Test that all vector layers have the correct CRS set
    @pytest.mark.parametrize('layer', ['damage', 'aoi', 'centroids'])
    def test_out_crs(self, output_path, layer):
        with fiona.open(output_path.joinpath('vector/damage.gpkg'), layer=layer) as src:
            assert src.crs == {'init': 'epsg:32615'}

    # Test for correct number of vector layers
    def test_out_file_layers(self, output_path):
        assert len(fiona.listlayers(output_path.joinpath('vector/damage.gpkg'))) == 3

    # Test output rasters for correct CRS
    @pytest.mark.parametrize('file,expected', [
        pytest.param('mosaics/overlay.tif', 32615, id='overlay_mos_crs'),
        pytest.param('mosaics/pre.tif', 32615, id='pre_mos_crs'),
        pytest.param('mosaics/post.tif', 32615, id='post_mos_crs'),
    ])
    def test_out_epsg(self, output_path, file, expected):
        with rasterio.open(output_path.joinpath(file)) as src:
            assert src.crs.to_epsg() == expected


class TestNoCUDA:

    @pytest.fixture(scope='class', autouse=True)
    def setup(self, staging_path, output_path):
        # Pass args to handler
        self.monkeypatch.setattr('argparse.ArgumentParser.parse_args', lambda x: Args(
            staging_path=staging_path,
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
    def setup(self, staging_path, output_path):
        # Pass args to handler
        self.monkeypatch.setattr('argparse.ArgumentParser.parse_args', lambda x: Args(
            staging_path=staging_path,
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
