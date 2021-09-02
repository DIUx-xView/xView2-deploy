from pathlib import Path
import pytest
import rasterio
import rasterio.crs
import handler
import utils.dataframe
from .conftest import Args


class TestMakeFootprintDF:

    def test_footprint_df(self):
        args = Args(destination_crs=rasterio.crs.CRS.from_epsg(32615))
        files = handler.get_files(Path('tests/data/input/pre'))
        df = utils.dataframe.make_footprint_df(files)
        assert df.shape == (4, 7)

    def test_footprint_df_crs_from_raster(self):
        args = Args(destination_crs=rasterio.crs.CRS.from_epsg(32615))
        files = handler.get_files(Path('tests/data/input/pre'))
        df = utils.dataframe.make_footprint_df(files)
        assert df.crs == rasterio.crs.CRS.from_epsg(26915)


# Todo: Fix these with fixtures file and parameters...may not even need this anymore
class TestGetTransformRes:

    def test_get_transform_res_from_src(self):
        args = Args(destination_crs=rasterio.crs.CRS.from_epsg(32615))
        file = 'tests/data/input/pre/tile_337-9136.tif'
        src = rasterio.open(file)
        test = utils.dataframe.get_trans_res(src.crs, src.width, src.height, src.bounds, args.destination_crs)
        assert test == pytest.approx((0.6000000000019269, 0.6000000000019269))

    def test_get_transform_res_from_argument(self):
        args = Args(pre_crs=rasterio.crs.CRS.from_epsg(26915), destination_crs=rasterio.crs.CRS.from_epsg(32615))
        file = 'tests/data/misc/no_crs/may24C350000e4102500n.jpg'
        src = rasterio.open(file)
        test = utils.dataframe.get_trans_res(args.pre_crs, src.width, src.height, src.bounds, args.destination_crs)
        assert test == pytest.approx((0.2500000000010606, 0.2500000000010606))

    def test_get_transform_res_from_src_with_argument(self):
        # Should use src CRS
        args = Args(pre_crs=rasterio.crs.CRS.from_epsg(26916), destination_crs=rasterio.crs.CRS.from_epsg(32615))
        file = 'tests/data/input/pre/tile_337-9136.tif'
        src = rasterio.open(file)
        test = utils.dataframe.get_trans_res(args.pre_crs, src.width, src.height, src.bounds, args.destination_crs)
        assert test == pytest.approx((0.6380336196569164, 0.6380336196569164))

    def test_transform_res_not_set_not_passed(self):
        pass

    def test_raster_not_georeferenced(self):
        pass


class TestGetUTM:

    def test_get_utm(self, pre_df):
        assert utils.dataframe.get_utm(pre_df) == 32615



class TestProcessDF:

    def test_process(self):
        args = Args(destination_crs=rasterio.crs.CRS.from_epsg(32615))
        files = handler.get_files(Path('tests/data/input/pre'))
        df = utils.dataframe.make_footprint_df(files)
        test = utils.dataframe.process_df(df, args.destination_crs)
        assert test.shape == (4, 8)


class TestGetIntersect:

    def test_get_intersect(self, pre_df, post_df):
        args = Args(destination_crs=rasterio.crs.CRS.from_epsg(26915))
        assert utils.dataframe.get_intersect(pre_df, post_df, args) == pytest.approx((366682.809231145, 4103282.4, 367871.4, 4104256.849245705))

    def test_not_rectangle(self, pre_df, post_df):
        args = Args(destination_crs=rasterio.crs.CRS.from_epsg(26915))
        assert utils.dataframe.get_intersect(pre_df[:3], post_df, args) == pytest.approx(366682.809231145, 4103282.4, 367871.4, 4104256.849245705)

    def test_dont_intersect(self, pre_df, no_intersect_df):
        args = Args(destination_crs=rasterio.crs.CRS.from_epsg(26915))
        with pytest.raises(AssertionError):
            assert utils.dataframe.get_intersect(pre_df, no_intersect_df, args)


class TestGetMaxRes:

    def test_max_res(self, pre_df, post_df):
        assert utils.dataframe.get_max_res(pre_df, post_df) == pytest.approx((0.5437457393600895, 0.6000000000019269))