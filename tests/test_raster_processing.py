import rasterio
import rasterio.crs
from pathlib import Path
from .conftest import Args
from utils import raster_processing, dataframe
import handler
import numpy as np


class TestCheckDims:

    def test_check_dims_full_size(self):
        with rasterio.open(Path('tests/data/output/chips/post/0_post.tif')) as src:
            arr = src.read()
        result = raster_processing.check_dims(arr, 1024, 1024)
        assert result.shape[1] == 1024
        assert result.shape[1] == 1024

    def test_check_dims_with_pad(self):
        with rasterio.open(Path('tests/data/output/chips/post/0_post.tif')) as src:
            arr = src.read()
        result = raster_processing.check_dims(arr, 1500, 1500)
        assert result.shape[1] == 1500
        assert result.shape[1] == 1500


class TestCreateMosaic:

    def test_create_mosaic(self, tmp_path, pre_df):
        args = Args(destination_crs=rasterio.crs.CRS.from_epsg(32615))
        out_path = tmp_path / 'out.tif'
        files_str = [str(file) for file in pre_df.filename]
        test = raster_processing.create_mosaic(files_str, out_path, pre_df.crs, args.destination_crs, (366642.6, 4103282.4, 367871.4, 4104511.2),(.6, .6))
        assert test.is_file()

    def test_create_mosaic_none_args(self, tmp_path, pre_df):
        args = Args(destination_crs=rasterio.crs.CRS.from_epsg(32615))
        out_path = tmp_path / 'out.tif'
        files_str = [str(file) for file in pre_df.filename]
        test = raster_processing.create_mosaic(files_str, out_path, None, None, None, (0.6, 0.6))
        assert test.is_file()


class TestCreatChips:

    def test_create_chips(self, tmp_path):

        print(tmp_path)
        out_dir = tmp_path / 'chips'
        out_dir.mkdir()
        in_mosaic = Path('tests/data/output/mosaics/pre.tif')
        intersect = (366676.6736748844, 4103281.8940772624, 367871.4000008028, 4104256.849355083)
        chips = raster_processing.create_chips(in_mosaic, out_dir, intersect)

        assert len(list(out_dir.iterdir())) == 4
        with rasterio.open(list(out_dir.iterdir())[0]) as src:
            assert src.height == 1024
            assert src.width == 1024


class TestCreateComposite:

    def test_create_composite(self, tmp_path):
        in_file = 'tests/data/output/chips/pre/0_pre.tif'
        with rasterio.open(in_file) as src:
            transforms = src.profile
        out_file = tmp_path / 'composite.tif'
        dmg_arr = np.load(open('tests/data/misc/damage_arr/cls_0.npy', 'rb'))
        assert raster_processing.create_composite(in_file, dmg_arr, out_file, transforms) == out_file


class TestCreateVRT:

    def test_vrt(self, tmp_path):
        in_dir = 'tests/data/input/pre'
        files = handler.get_files(in_dir)
        out_path = tmp_path / 'vrt.vrt'
        test = raster_processing.create_vrt(files, out_path)
        assert out_path.is_file()


class TestGetRes:

    def test_get_image_res(self):
        assert raster_processing.get_res('tests/data/input/pre/tile_337-9136.tif') == (0.6, 0.6)



