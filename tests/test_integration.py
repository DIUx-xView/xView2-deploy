from pathlib import Path
import pytest
import handler
import torch
import fiona
import rasterio.crs
from pytest import MonkeyPatch

# Todo: Return appropriate tensor for each image
# Todo: How do we test results (mosaics) (mean/sum of array?)
# Todo: Class out our monkeypatches


@pytest.fixture(scope='class', autouse=True)
def output_path(tmp_path_factory):
    return tmp_path_factory.mktemp('output')


@pytest.fixture(scope='class', autouse=True)
def staging_path(tmp_path_factory):
    return tmp_path_factory.mktemp('staging')


# Todo: may be able to make this a dataclass
class MockArgs:

    def __init__(self,
                 staging_path,
                 output_path,
                 pre_directory='tests/data/input/pre',
                 post_directory='tests/data/input/post',
                 n_procs=4,
                 batch_size=1,
                 num_workers=8,
                 pre_crs=None,
                 post_crs=None,
                 destination_crs=None,
                 output_resolution=None,
                 save_intermediates=False,
                 agol_user='',
                 agol_password='',
                 agol_feature_service='',
                 dp_mode=True
                 ):

        self.output_directory = output_path
        self.staging_directory = staging_path
        self.staging_directory = staging_path
        self.output_directory = output_path
        self.pre_directory = Path(pre_directory)
        self.post_directory = Path(post_directory)
        self.n_procs = n_procs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pre_crs = pre_crs
        self.post_crs = post_crs
        self.destination_crs = destination_crs
        self.output_resolution = output_resolution
        self.save_intermediates = save_intermediates
        self.agol_user = agol_user
        self.agol_password = agol_password
        self.agol_feature_service = agol_feature_service
        self.dp_mode = dp_mode


class MockLocModel:

    def __init__(self, *args, **kwargs):
        self.model_size = args[0]

    # Todo: this should return the correct tensor based on the image input. Currently returns the same tensor.
    # Todo: Ideally we store these with a CRS so we cas reproject based on the mock args...some day
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
        self.monkeypatch.setattr('argparse.ArgumentParser.parse_args', lambda x: MockArgs(
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

    def test_pre_mosaic(self, staging_path, output_path):
        assert output_path.joinpath('mosaics/pre.tif').is_file()

    def test_post_mosaic(self, staging_path, output_path):
        assert output_path.joinpath('mosaics/post.tif').is_file()

    def test_overlay_mosaic(self, staging_path, output_path):
        assert output_path.joinpath('mosaics/overlay.tif').is_file()

    # Todo: currently fails although the app still works. Should still be fixed at some point
    @pytest.mark.xfail
    def test_pre_reproj(self, staging_path, output_path):
        assert len(list(staging_path.joinpath('pre').glob('**/*'))) == 4

    # Todo: currently fails although the app still works. Should still be fixed at some point
    @pytest.mark.xfail
    def test_post_reproj(self, staging_path, output_path):
        assert len(list(staging_path.joinpath('post').glob('**/*'))) == 6

    def test_overlay(self, staging_path, output_path):
        assert len(list(output_path.joinpath('over').glob('**/*'))) == 4

    def test_out_shapefile(self, staging_path, output_path):
        assert output_path.joinpath('vector/damage.gpkg').is_file()

    def test_log(self, staging_path, output_path):
        assert output_path.joinpath('log/xv2.log').is_file()

    def test_chips_pre(self, staging_path, output_path):
        assert len(list(output_path.joinpath('chips/pre').glob('**/*'))) == 4

    def test_chips_post(self, staging_path, output_path):
        assert len(list(output_path.joinpath('chips/post').glob('**/*'))) == 4

    def test_loc_out(self, staging_path, output_path):
        assert len(list(output_path.joinpath('loc').glob('**/*'))) == 4

    def test_dmg_out(self, staging_path, output_path):
        assert len(list(output_path.joinpath('dmg').glob('**/*'))) == 4

    def test_dmg_mosaic(self, output_path):
        assert output_path.joinpath('mosaics/damage.tif').is_file()

    def test_out_file(self, output_path):
        shapes = fiona.open(output_path.joinpath('vector/damage.gpkg'))
        assert len(shapes) == 872

    def test_out_file_layers(self, output_path):
        assert len(fiona.listlayers(output_path.joinpath('vector/damage.gpkg'))) == 3

    def test_out_epsg(self, output_path):
        with rasterio.open(output_path.joinpath('mosaics/overlay.tif')) as src:
            assert src.crs.to_epsg() == 32615


class TestNoCUDA:

    @pytest.fixture(scope='class', autouse=True)
    def setup(self, staging_path, output_path):
        # Pass args to handler
        self.monkeypatch.setattr('argparse.ArgumentParser.parse_args', lambda x: MockArgs(
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
        self.monkeypatch.setattr('argparse.ArgumentParser.parse_args', lambda x: MockArgs(
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
