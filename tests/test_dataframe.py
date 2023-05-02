from pathlib import Path
import fiona.errors
import geopandas
import pytest
import rasterio
import rasterio.crs
import handler
import utils.dataframe
from .conftest import Args


class TestMakeFootprintDF:
    def test_footprint_df(self):
        args = Args(destination_crs=rasterio.crs.CRS.from_epsg(32615))
        files = handler.get_files(Path("tests/data/input/pre"))
        df = utils.dataframe.make_footprint_df(files)
        assert df.shape == (4, 7)

    def test_footprint_df_crs_from_raster(self):
        args = Args(destination_crs=rasterio.crs.CRS.from_epsg(32615))
        files = handler.get_files(Path("tests/data/input/pre"))
        df = utils.dataframe.make_footprint_df(files)
        assert df.crs == rasterio.crs.CRS.from_epsg(26915)

    # Todo: finish test for no crs...not sure if this is working


class TestGetTransformRes:
    def test_get_transform_res_from_src(self):
        args = Args(destination_crs=rasterio.crs.CRS.from_epsg(32615))
        file = "tests/data/input/pre/tile_337-9136.tif"
        src = rasterio.open(file)
        test = utils.dataframe.get_trans_res(
            src.crs, src.width, src.height, src.bounds, args.destination_crs
        )
        assert test == pytest.approx(
            (0.6000000000019269, 0.6000000000019269), abs=1e-06
        )

    def test_get_transform_res_from_argument(self):
        args = Args(
            pre_crs=rasterio.crs.CRS.from_epsg(26915),
            destination_crs=rasterio.crs.CRS.from_epsg(32615),
        )
        file = "tests/data/misc/no_crs/may24C350000e4102500n.jpg"
        src = rasterio.open(file)
        test = utils.dataframe.get_trans_res(
            args.pre_crs, src.width, src.height, src.bounds, args.destination_crs
        )
        assert test == pytest.approx(
            (0.2500000000010606, 0.2500000000010606), abs=1e-06
        )

    def test_get_transform_res_from_src_with_argument(self):
        # Should use src CRS
        args = Args(
            pre_crs=rasterio.crs.CRS.from_epsg(26916),
            destination_crs=rasterio.crs.CRS.from_epsg(32615),
        )
        file = "tests/data/input/pre/tile_337-9136.tif"
        src = rasterio.open(file)
        test = utils.dataframe.get_trans_res(
            args.pre_crs, src.width, src.height, src.bounds, args.destination_crs
        )
        assert test == pytest.approx(
            (0.6380336196569164, 0.6380336196569164), abs=1e-06
        )

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
        files = handler.get_files(Path("tests/data/input/pre"))
        df = utils.dataframe.make_footprint_df(files)
        test = utils.dataframe.process_df(df, args.destination_crs)
        assert test.shape == (4, 8)


class TestGetIntersect:
    @pytest.mark.parametrize(
        "poly_name,expected",
        [
            pytest.param(
                ["within"], (366857, 4103474, 367772, 4104074), id="int_rect_within"
            ),
            pytest.param(
                ["within", "outside"],
                (366857, 4103474, 367772, 4104074),
                id="int_rect_within_out",
            ),
            pytest.param(
                ["intersects"],
                (367626, 4103895, 367871, 4104242),
                id="int_rect_intersects",
            ),
            pytest.param(
                ["intersects", "outside"],
                (367626, 4103895, 367871, 4104242),
                id="int_rect_intersects_out",
            ),
            pytest.param(
                ["intersects", "within"],
                (366857, 4103474, 367871, 4104242),
                id="int_rect_intersects_within",
            ),
        ],
    )
    def test_get_intersect(self, pre_df, post_df, aoi_df, poly_name, expected):
        args = Args(destination_crs=rasterio.crs.CRS.from_epsg(26915))
        aoi = aoi_df[aoi_df.name.isin(poly_name)]
        assert utils.dataframe.get_intersect(
            pre_df, post_df, args, aoi
        ).bounds == pytest.approx(expected, abs=2)

    def test_intersect_fail(self, pre_df, no_intersect_df):
        args = Args(destination_crs=rasterio.crs.CRS.from_epsg(26915))
        with pytest.raises(AssertionError):
            assert utils.dataframe.get_intersect(pre_df, no_intersect_df, args)

    def test_int_aoi_intersect(self, pre_df, post_df, aoi_df):
        args = Args(destination_crs=rasterio.crs.CRS.from_epsg(26915))
        aoi = aoi_df[aoi_df.name.isin(["outside"])]
        with pytest.raises(AssertionError):
            assert utils.dataframe.get_intersect(pre_df, post_df, args, aoi)

    def test_int_bldg_poly(self, pre_df, bldg_poly_df, post_df):
        args = Args(destination_crs=rasterio.crs.CRS.from_epsg(26915))
        intersect = utils.dataframe.get_intersect(
            pre_df, post_df, args, aoi=None, in_poly_df=bldg_poly_df
        )
        assert intersect.bounds == pytest.approx(
            (366752, 4103766, 367241, 4104183), abs=2
        )

    def test_int_bldg_poly_new_please_param_me(self, pre_df, post_df):
        args = Args(destination_crs=rasterio.crs.CRS.from_epsg(26915))
        bldg_poly_df = geopandas.read_file("tests/data/misc/joplin.geojson")
        intersect = utils.dataframe.get_intersect(
            pre_df, post_df, args, aoi=None, in_poly_df=bldg_poly_df
        )
        assert intersect.bounds == pytest.approx(
            (366682, 4103282, 367871, 4104257), abs=2
        )

    def test_int_no_aoi_no_bldgs(self, pre_df, bldg_poly_df, post_df):
        args = Args(destination_crs=rasterio.crs.CRS.from_epsg(26915))
        intersect = utils.dataframe.get_intersect(pre_df, post_df, args)
        assert intersect.bounds == pytest.approx(
            (366682, 4103282, 367871, 4104256), abs=2
        )


class TestGetMaxRes:
    def test_max_res(self, pre_df, post_df):
        assert utils.dataframe.get_max_res(pre_df, post_df) == pytest.approx(
            (0.6000000000032912, 0.6739616404124325)
        )


class TestBldgPolyProcess:
    def test_bldg_poly_process(self, bldg_poly_df, tmpdir):
        args = Args(destination_crs=rasterio.crs.CRS.from_epsg(26915))
        intersect = (
            366682.8771564872,
            4103280.987792889,
            367872.6850531151,
            4104257.0333117004,
        )
        with rasterio.open("tests/data/output/mosaics/post.tif") as src:
            out_shape = (src.height, src.width)
            out_transform = src.transform
        out_file = f"{tmpdir}/bldg_mosaic.tif"
        test = utils.dataframe.bldg_poly_process(
            bldg_poly_df,
            intersect,
            args.destination_crs,
            out_file,
            out_shape,
            out_transform,
        )
        assert Path(out_file).is_file()
