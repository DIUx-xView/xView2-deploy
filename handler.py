import platform
import timeit
import argparse
import os
import sys
import multiprocessing as mp

mp.set_start_method("spawn", force=True)
import utils.dataframe
import numpy as np
from utils import raster_processing, features, dataframe
import rasterio.warp
import rasterio.crs
import torch
from collections import defaultdict
from os import makedirs, path
from pathlib import Path
from torch.utils.data import DataLoader
from skimage.morphology import square, dilation
from tqdm import tqdm
from dataset import XViewDataset
from models import XViewFirstPlaceLocModel, XViewFirstPlaceClsModel
from loguru import logger
from osgeo import gdal
import shapely


class Options(object):
    def __init__(
        self,
        pre_path="input/pre",
        post_path="input/post",
        poly_chip=None,
        out_loc_path="output/loc",
        out_dmg_path="output/dmg",
        out_overlay_path="output/over",
        model_config="configs/model.yaml",
        model_weights="weights/weight.pth",
        geo_profile=None,
        use_gpu=False,
        vis=False,
    ):
        self.in_pre_path = pre_path
        self.in_post_path = post_path
        self.poly_chip = poly_chip
        self.out_loc_path = out_loc_path
        self.out_cls_path = out_dmg_path
        self.out_overlay_path = out_overlay_path
        self.model_config_path = model_config
        self.model_weight_path = model_weights
        self.geo_profile = geo_profile
        self.is_use_gpu = use_gpu
        self.is_vis = vis


class Files(object):
    def __init__(
        self,
        ident,
        pre_directory,
        post_directory,
        output_directory,
        pre,
        post,
        poly_chip,
    ):
        self.ident = ident
        self.pre = pre_directory.joinpath(
            pre
        ).resolve()  # Todo: These don't seem right...why are we joining to the pre_directory?
        self.post = post_directory.joinpath(post).resolve()
        if poly_chip:
            poly_chip = poly_chip.joinpath(poly_chip).resolve()
        self.poly_chip = poly_chip
        self.loc = (
            output_directory.joinpath("loc").joinpath(f"{self.ident}.tif").resolve()
        )
        self.dmg = (
            output_directory.joinpath("dmg").joinpath(f"{self.ident}.tif").resolve()
        )
        self.over = (
            output_directory.joinpath("over").joinpath(f"{self.ident}.tif").resolve()
        )
        self.profile = self.get_profile()
        self.transform = self.profile["transform"]
        self.opts = Options(
            pre_path=self.pre,
            post_path=self.post,
            poly_chip=self.poly_chip,
            out_loc_path=self.loc,
            out_dmg_path=self.dmg,
            out_overlay_path=self.over,
            geo_profile=self.profile,
            vis=True,
            use_gpu=True,
        )

    def get_profile(self):
        with rasterio.open(self.pre) as src:
            return src.profile


def make_output_structure(output_path):
    """
    Creates directory structure for outputs.
    :param output_path: Output path
    :return: True if succussful
    """

    Path(f"{output_path}/mosaics").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/chips/pre").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/chips/post").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/chips/in_polys").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/loc").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/dmg").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/over").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/vector").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/vrt").mkdir(parents=True, exist_ok=True)

    return True


def get_files(dirname, extensions=[".png", ".tif", ".jpg"]):
    """
    Gathers list of files for processing from path recursively.
    :param dirname: path to parse
    :param extensions: extensions to match
    :return: list of files matching extensions
    """
    dir_path = Path(dirname).resolve()

    files = dir_path.glob("**/*")

    match = [path.resolve() for path in files if path.suffix in extensions]

    assert len(match) > 0, logger.critical(
        f"No image files found in {dir_path.resolve()}"
    )
    logger.debug(f"Retrieved {len(match)} files from {dirname}")

    return match


def reproject_helper(args, raster_tuple, procnum, return_dict, resolution):
    """
    Helper function for reprojection
    """
    (pre_post, src_crs, raster_file) = raster_tuple
    basename = raster_file.stem
    dest_file = args.staging_directory.joinpath("pre").joinpath(f"{basename}.tif")
    try:
        return_dict[procnum] = (
            pre_post,
            raster_processing.reproject(
                raster_file, dest_file, src_crs, args.destination_crs, resolution
            ),
        )
    except ValueError:
        return None


def postprocess_and_write(result_dict):
    """
    Postprocess results from inference and write results to file
    :param result_dict: dictionary containing all required opts for each example
    """
    _thr = [0.38, 0.13, 0.14]
    pred_coefs = [1.0] * 4  # not 12, b/c already took mean over 3 in each subset
    loc_coefs = [1.0] * 4

    preds = []
    _i = -1
    for k, v in result_dict.items():
        if "cls" in k:
            _i += 1
            # Todo: I think the below can just be replaced by v['cls'] -- should check
            msk = v["cls"].numpy()
            preds.append(msk * pred_coefs[_i])

    preds = np.asarray(preds).astype("float").sum(axis=0) / np.sum(pred_coefs) / 255
    msk_dmg = preds[..., 1:].argmax(axis=2) + 1

    if v["poly_chip"] == "None":  # Must be a string or PyTorch throws an error
        loc_preds = []
        _i = -1
        for k, v in result_dict.items():
            if "loc" in k:
                _i += 1
                msk = v["loc"].numpy()
                loc_preds.append(msk * loc_coefs[_i])

        loc_preds = (
            np.asarray(loc_preds).astype("float").sum(axis=0) / np.sum(loc_coefs) / 255
        )
        msk_loc = (
            1
            * (
                (loc_preds > _thr[0])
                | ((loc_preds > _thr[1]) & (msk_dmg > 1) & (msk_dmg < 4))
                | ((loc_preds > _thr[2]) & (msk_dmg > 1))
            )
        ).astype("uint8")
    else:
        msk_loc = rasterio.open(v["poly_chip"]).read(1)

    msk_dmg = msk_dmg * msk_loc
    _msk = msk_dmg == 2
    if _msk.sum() > 0:
        _msk = dilation(_msk, square(5))
        msk_dmg[_msk & msk_dmg == 1] = 2

    msk_dmg = msk_dmg.astype("uint8")

    loc = msk_loc
    cls = msk_dmg

    sample_result_dict = result_dict["34cls"]
    sample_result_dict["geo_profile"].update(dtype=rasterio.uint8)

    dst = rasterio.open(
        sample_result_dict["out_loc_path"], "w", **sample_result_dict["geo_profile"]
    )
    dst.write(loc, 1)
    dst.close()

    dst = rasterio.open(
        sample_result_dict["out_cls_path"], "w", **sample_result_dict["geo_profile"]
    )
    dst.write(cls, 1)
    dst.close()

    if sample_result_dict["is_vis"]:
        raster_processing.create_composite(
            sample_result_dict["in_pre_path"],
            cls,
            sample_result_dict["out_overlay_path"],
            sample_result_dict["geo_profile"],
        )


def run_inference(
    loader, model_wrapper, write_output=False, mode="loc", return_dict=None
):
    results = defaultdict(list)
    with torch.no_grad():  # This is really important to not explode memory with gradients!
        for ii, result_dict in tqdm(enumerate(loader), total=len(loader)):
            # print(result_dict['in_pre_path'])
            debug = False
            # if '116' in result_dict['in_pre_path'][0]:
            #    import ipdb; ipdb.set_trace()
            #    debug=True

            out = model_wrapper.forward(result_dict["img"], debug=debug)

            # Save prediction tensors for testing
            # Todo: Create argument for turning on debug/trace/test data
            # pred_path = Path('/home/ubuntu/debug/output/preds')
            # makedirs(pred_path, exist_ok=True)
            # torch.save(out, pred_path / f'preds_{mode}_{ii}.pt')

            out = out.detach().cpu()

            del result_dict["img"]

            if "pre_image" in result_dict:
                result_dict["pre_image"] = result_dict["pre_image"].cpu().numpy()
            if "post_img" in result_dict:
                result_dict["post_image"] = result_dict["post_image"].cpu().numpy()
            if mode == "loc":
                result_dict["loc"] = out
            elif mode == "cls":
                result_dict["cls"] = out
            else:
                raise ValueError("Incorrect mode -- must be loc or cls")
            # Do this one separately because you can't return a class from a dataloader
            result_dict["geo_profile"] = [
                loader.dataset.pairs[idx].opts.geo_profile for idx in result_dict["idx"]
            ]
            for k, v in result_dict.items():
                results[k] = results[k] + list(v)

    # Making a list
    results_list = [dict(zip(results, t)) for t in zip(*results.values())]
    if write_output:
        import cv2  # Moved here to prevent import error

        pred_folder = args.output_directory / "preds"
        logger.info("Writing results...")
        makedirs(pred_folder, exist_ok=True)
        for result in tqdm(results_list, total=len(results_list)):
            # TODO: Multithread this to make it more efficient/maybe eliminate it from workflow
            if mode == "loc":
                cv2.imwrite(
                    path.join(
                        pred_folder,
                        result["in_pre_path"]
                        .split("/")[-1]
                        .replace(".tif", "_part1.png"),
                    ),
                    np.array(result["loc"])[...],
                    [cv2.IMWRITE_PNG_COMPRESSION, 9],
                )
            elif mode == "cls":
                cv2.imwrite(
                    path.join(
                        pred_folder,
                        result["in_pre_path"]
                        .split("/")[-1]
                        .replace(".tif", "_part1.png"),
                    ),
                    np.array(result["cls"])[..., :3],
                    [cv2.IMWRITE_PNG_COMPRESSION, 9],
                )
                cv2.imwrite(
                    path.join(
                        pred_folder,
                        result["in_pre_path"]
                        .split("/")[-1]
                        .replace(".tif", "_part2.png"),
                    ),
                    np.array(result["cls"])[..., 2:],
                    [cv2.IMWRITE_PNG_COMPRESSION, 9],
                )
    if return_dict is None:
        return results_list
    else:
        return_dict[f"{model_wrapper.model_size}{mode}"] = results_list


# Todo: Move this to raster_processing
def check_data(images):
    """
    Check that our image pairs contain useful data. Note: This only check the first band of each file.
    :param images: Images to check for data
    :return: True if both images contain useful data. False if either contains no useful date.
    """
    for image in images:
        src = rasterio.open(image)
        layer = src.read(1)
        src.close()
        if layer.sum() == 0:
            return False

    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create arguments for xView 2 handler."
    )

    parser.add_argument(
        "--pre_directory",
        metavar="/path/to/pre/files/",
        type=Path,
        required=True,
        help="Directory containing pre-disaster imagery. This is searched recursively.",
    )
    parser.add_argument(
        "--post_directory",
        metavar="/path/to/post/files/",
        type=Path,
        required=True,
        help="Directory containing post-disaster imagery. This is searched recursively.",
    )
    parser.add_argument(
        "--output_directory",
        metavar="/path/to/output/",
        type=Path,
        required=True,
        help="Directory to store output files. This will be created if it does not exist. Existing files may be overwritten.",
    )
    parser.add_argument(
        "--n_procs",
        default=8,
        help="Number of processors for multiprocessing",
        type=int,
    )
    parser.add_argument(
        "--batch_size",
        default=2,
        help="Number of chips to run inference on at once",
        type=int,
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        help="Number of workers loading data into RAM. Recommend 4 * num_gpu",
        type=int,
    )
    parser.add_argument(
        "--pre_crs",
        help='The Coordinate Reference System (CRS) for the pre-disaster imagery. This will only be utilized if images lack CRS data. May be WKT, EPSG (ex. "EPSG:4326"), or PROJ string.',
    )
    parser.add_argument(
        "--post_crs",
        help='The Coordinate Reference System (CRS) for the post-disaster imagery. This will only be utilized if images lack CRS data. May be WKT, EPSG (ex. "EPSG:4326"), or PROJ string.',
    )
    parser.add_argument(
        "--destination_crs",
        default=None,
        help='The Coordinate Reference System (CRS) for the output overlays. May be WKT, EPSG (ex. "EPSG:4326"), or PROJ string. Leave blank to calculate the approriate UTM zone.',
    )  # Todo: Create warning/force change when not using a CRS that utilizes meters for base units
    parser.add_argument(
        "--dp_mode",
        default=False,
        action="store_true",
        help="Run models serially, but using DataParallel",
    )
    parser.add_argument(
        "--output_resolution",
        default=None,
        help="Override minimum resolution calculator. This should be a lower resolution (higher number) than source imagery for decreased inference time. Must be in units of destinationCRS.",
    )
    parser.add_argument(
        "--save_intermediates",
        default=False,
        action="store_true",
        help="Store intermediate runfiles",
    )
    parser.add_argument(
        "--aoi_file", default=None, help="Shapefile or GeoJSON file of AOI polygons"
    )
    parser.add_argument(
        "--bldg_polys",
        default=None,
        help="Shapefile or GeoJSON file of input building footprints",
    )

    return parser.parse_args()


def pre_post_handler(args, pre_post):
    if pre_post == "pre":
        crs_arg = args.pre_crs
        directory = args.pre_directory
    elif pre_post == "post":
        crs_arg = args.post_crs
        directory = args.post_directory
    else:
        raise ValueError("pre_post must be either pre or post.")

    crs = rasterio.crs.CRS.from_string(crs_arg) if crs_arg else None
    files = get_files(directory)
    df = utils.dataframe.make_footprint_df(files)

    return df, crs


@logger.catch()
def main():
    t0 = timeit.default_timer()

    # Make file structure
    make_output_structure(args.output_directory)

    # Create post df and determine crs
    pre_df, args.pre_crs = pre_post_handler(args, "pre")
    post_df, args.post_crs = pre_post_handler(args, "post")

    # Create destination CRS object from argument, else determine UTM zone and create CRS object
    dest_crs = utils.dataframe.get_utm(pre_df)
    logger.info(f"Calculated CRS: {dest_crs}")

    if args.destination_crs:
        args.destination_crs = rasterio.crs.CRS.from_string(args.destination_crs)
        logger.info(
            f"Calculated CRS overridden by passed argument: {args.destination_crs}"
        )
    else:
        args.destination_crs = dest_crs

    # Ensure CRS is projected. This prevents a lot of problems downstream.
    assert args.destination_crs.is_projected, logger.critical(
        "CRS is not projected. Please use a projected CRS"
    )

    # Process DF
    pre_df = utils.dataframe.process_df(pre_df, args.destination_crs)
    post_df = utils.dataframe.process_df(post_df, args.destination_crs)

    if args.bldg_polys:
        in_poly_df = dataframe.bldg_poly_handler(args.bldg_polys).to_crs(
            args.destination_crs
        )
    else:
        in_poly_df = None

    # Get AOI files and calculate intersect
    if args.aoi_file:
        aoi_df = dataframe.make_aoi_df(args.aoi_file).to_crs(args.destination_crs)
    else:
        aoi_df = None

    extent = utils.dataframe.get_intersect(
        pre_df, post_df, args, aoi_df, in_poly_df
    )  # Todo: should probably pass back the shape and call the bounds when needed
    logger.info(f"Calculated extent: {extent.bounds}")

    # Calculate destination resolution
    res = dataframe.get_max_res(pre_df, post_df)
    logger.info(f"Calculated resolution: {res}")

    if args.output_resolution:
        res = (args.output_resolution, args.output_resolution)
        logger.info(f"Calculated resolution overridden by passed argument: {res}")

    logger.info("Creating pre mosaic...")
    pre_mosaic = raster_processing.create_mosaic(
        [str(file) for file in pre_df.filename],
        Path(f"{args.output_directory}/mosaics/pre.tif"),
        pre_df.crs,
        args.destination_crs,
        extent.bounds,
        res,
        aoi_df,
    )

    logger.info("Creating post mosaic...")
    post_mosaic = raster_processing.create_mosaic(
        [str(file) for file in post_df.filename],
        Path(f"{args.output_directory}/mosaics/post.tif"),
        post_df.crs,
        args.destination_crs,
        extent.bounds,
        res,
        aoi_df,
    )

    if args.bldg_polys:
        logger.info("Creating input polygon mosaic...")
        with rasterio.open(post_mosaic) as src:
            out_shape = (src.height, src.width)
            out_transform = src.transform

        in_poly_mosaic = dataframe.bldg_poly_process(
            in_poly_df,
            extent.bounds,
            args.destination_crs,
            Path(f"{args.output_directory}/mosaics/in_polys.tif"),
            out_shape,
            out_transform,
        )

    logger.info("Chipping...")
    pre_chips = raster_processing.create_chips(
        pre_mosaic,
        args.output_directory.joinpath("chips").joinpath("pre"),
        extent.bounds,
    )
    logger.debug(f"Num pre chips: {len(pre_chips)}")
    post_chips = raster_processing.create_chips(
        post_mosaic,
        args.output_directory.joinpath("chips").joinpath("post"),
        extent.bounds,
    )
    logger.debug(f"Num post chips: {len(post_chips)}")

    if args.bldg_polys:
        poly_chips = raster_processing.create_chips(
            in_poly_mosaic,
            args.output_directory.joinpath("chips").joinpath("in_polys"),
            extent.bounds,
        )

        assert len(pre_chips) == len(poly_chips), logger.error(
            "Chip numbers mismatch (in polys"
        )
    else:
        poly_chips = [None] * len(pre_chips)

    assert len(pre_chips) == len(post_chips), logger.error(
        "Chip numbers mismatch (pre/post"
    )

    # Defining dataset and dataloader
    pairs = []
    for idx, (pre, post, poly) in enumerate(zip(pre_chips, post_chips, poly_chips)):
        if not check_data([pre, post]):
            continue

        pairs.append(
            Files(
                pre.stem,
                args.pre_directory,
                args.post_directory,
                args.output_directory,
                pre,
                post,
                poly,
            )
        )

    eval_loc_dataset = XViewDataset(pairs, "loc")
    eval_loc_dataloader = DataLoader(
        eval_loc_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    eval_cls_dataset = XViewDataset(pairs, "cls")
    eval_cls_dataloader = DataLoader(
        eval_cls_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    # Todo: If on a one GPU machine (or other unsupported GPU count), force DP mode
    if torch.cuda.device_count() == 1:
        args.dp_mode = True
        logger.info("Single CUDA device found. Forcing DP mode")

    # Inference in DP mode
    if args.dp_mode:
        results_dict = {}

        for sz in ["34", "50", "92", "154"]:
            logger.info(f"Running models of size {sz}...")

            return_dict = {}
            loc_wrapper = XViewFirstPlaceLocModel(sz, dp_mode=args.dp_mode)

            if not args.bldg_polys:
                run_inference(
                    eval_loc_dataloader,
                    loc_wrapper,
                    args.save_intermediates,
                    "loc",
                    return_dict,
                )

            del loc_wrapper

            cls_wrapper = XViewFirstPlaceClsModel(sz, dp_mode=args.dp_mode)

            run_inference(
                eval_cls_dataloader,
                cls_wrapper,
                args.save_intermediates,
                "cls",
                return_dict,
            )

            del cls_wrapper

            results_dict.update({k: v for k, v in return_dict.items()})

    else:
        if torch.cuda.device_count() == 2:
            # For 2-GPU machines [TESTED]

            # Loading model
            loc_gpus = {
                "34": [0, 0, 0],
                "50": [1, 1, 1],
                "92": [0, 0, 0],
                "154": [1, 1, 1],
            }

            cls_gpus = {
                "34": [1, 1, 1],
                "50": [0, 0, 0],
                "92": [1, 1, 1],
                "154": [0, 0, 0],
            }

        elif torch.cuda.device_count() == 4:
            # For 4-GPU machines [ Todo: Test ]

            # Loading model
            loc_gpus = {
                "34": [0, 0, 0],
                "50": [1, 1, 1],
                "92": [2, 2, 2],
                "154": [3, 3, 3],
            }

            cls_gpus = {
                "34": [0, 0, 0],
                "50": [1, 1, 1],
                "92": [2, 2, 2],
                "154": [3, 3, 3],
            }

        elif torch.cuda.device_count() == 8:
            # For 8-GPU machines

            # Loading model
            loc_gpus = {
                "34": [0, 0, 0],
                "50": [1, 1, 1],
                "92": [2, 2, 2],
                "154": [3, 3, 3],
            }

            cls_gpus = {
                "34": [4, 4, 4],
                "50": [5, 5, 5],
                "92": [6, 6, 6],
                "154": [7, 7, 7],
            }

        else:
            raise ValueError(
                "Unsupported number of GPUs. Please use a 1, 2, 4, or 8 GPUs."
            )

        results_dict = {}

        # Running inference
        logger.info("Running inference...")

        for sz in loc_gpus.keys():
            logger.info(f"Running models of size {sz}...")
            loc_wrapper = XViewFirstPlaceLocModel(sz, devices=loc_gpus[sz])
            cls_wrapper = XViewFirstPlaceClsModel(sz, devices=cls_gpus[sz])

            # Running inference
            logger.info("Running inference...")

            # Run inference in parallel processes
            manager = mp.Manager()
            return_dict = manager.dict()
            jobs = []

            # Launch multiprocessing jobs for different pytorch jobs
            p1 = mp.Process(
                target=run_inference,
                args=(
                    eval_cls_dataloader,
                    cls_wrapper,
                    args.save_intermediates,
                    "cls",
                    return_dict,
                ),
            )
            p2 = mp.Process(
                target=run_inference,
                args=(
                    eval_loc_dataloader,
                    loc_wrapper,
                    args.save_intermediates,
                    "loc",
                    return_dict,
                ),
            )
            p1.start()

            if not args.bldg_polys:
                p2.start()

            jobs.append(p1)

            if not args.bldg_polys:
                jobs.append(p2)

            for proc in jobs:
                proc.join()

            results_dict.update({k: v for k, v in return_dict.items()})

        raise ValueError(logger.critical("Must use either 2, 4, or 8 GPUs."))

    # Quick check to make sure the samples in cls and loc are in the same order
    # assert(results_dict['34loc'][4]['in_pre_path'] == results_dict['34cls'][4]['in_pre_path'])

    results_list = [
        {k: v[i] for k, v in results_dict.items()}
        for i in range(len(results_dict["34cls"]))
    ]

    # Running postprocessing
    p = mp.Pool(args.n_procs)
    # postprocess_and_write(results_list[0])
    f_p = postprocess_and_write
    p.map(f_p, results_list)

    # Get files for creating vector file
    logger.info("Generating vector data")
    dmg_files = get_files(Path(args.output_directory) / "dmg")

    # if not using input polys use threshold to filter out small polygons (likely false positives)
    if args.bldg_polys:
        polygons = features.create_polys(dmg_files, threshold=0)
    else:
        polygons = features.create_polys(dmg_files)
        
    polygons = polygons

    if args.bldg_polys:
        polygons = in_poly_df.reset_index().overlay(polygons, how="identity").clip(extent)
        polygons = (
            polygons.groupby("index", as_index=False)
            .apply(lambda x: features.weight_dmg(x, args.destination_crs))
        )
        polygons.set_crs(args.destination_crs)

    polygons.geometry = polygons.geometry.simplify(1)

    # Create geojson
    json_out = Path(args.output_directory).joinpath("vector") / "damage.geojson"
    polygons.to_file(json_out, driver="GeoJSON", index=False)

    aoi = features.create_aoi_poly(polygons)
    centroids = features.create_centroids(polygons)

    logger.info(f"Polygons created: {len(polygons)}")
    logger.info(
        f"Inferred hull area: {aoi.area}"
    )  # Todo: Calculate area for pre/post/poly/aoi/intersect (the one that matters)

    # Create geopackage
    logger.info("Writing output file")
    vector_out = Path(args.output_directory).joinpath("vector") / "damage.gpkg"
    features.write_output(
        polygons, vector_out, layer="damage"
    )  # Todo: move this up to right after the polys are simplified to capture some vector data if script crashes
    features.write_output(aoi, vector_out, "aoi")
    features.write_output(centroids, vector_out, "centroids")

    # Create damage and overlay mosaics
    # Probably stop generating damage mosaic and create overlay from pre and vectors. Stop making overlay from chips
    logger.info("Creating damage mosaic")
    dmg_path = Path(args.output_directory) / "dmg"
    damage_files = [x for x in get_files(dmg_path)]
    damage_mosaic = raster_processing.create_mosaic(
        [str(file) for file in damage_files],
        Path(f"{args.output_directory}/mosaics/damage.tif"),
        None,
        None,
        None,
        res,
    )

    logger.info("Creating overlay mosaic")
    p = Path(args.output_directory) / "over"
    overlay_files = [x for x in get_files(p)]
    overlay_mosaic = raster_processing.create_mosaic(
        [str(file) for file in overlay_files],
        Path(f"{args.output_directory}/mosaics/overlay.tif"),
        None,
        None,
        None,
        res,
    )

    # Complete
    elapsed = timeit.default_timer() - t0
    logger.success(f"Run complete in {elapsed / 60:.3f} min")


def init():
    # Todo: Fix this global at some point
    global args
    args = parse_args()

    # Configure our logger and push our inputs
    # Todo: Capture sys info (gpu, procs, etc)
    logger.remove()
    logger.configure(
        handlers=[
            # Todo: Add argument to change log level
            dict(sink=sys.stderr, format="[{level}] {message}", level="INFO"),
            dict(
                sink=args.output_directory / "log" / f"xv2.log",
                enqueue=True,
                level="DEBUG",
                backtrace=True,
            ),
        ],
    )
    logger.opt(exception=True)

    # Scrub args of AGOL username and password and log them for debugging
    clean_args = {
        k: v
        for (k, v) in args.__dict__.items()
        if k != "agol_password"
        if k != "agol_user"
    }
    logger.debug(f"Run from:{__file__}")
    for k, v in clean_args.items():
        logger.debug(f"{k}: {v}")

    # Add system info to log
    logger.debug(f"System: {platform.platform()}")
    logger.debug(f"Python: {sys.version}")
    logger.debug(f"Torch: {torch.__version__}")
    logger.debug(f"GDAL: {gdal.__version__}")
    logger.debug(f"Rasterio: {rasterio.__version__}")
    logger.debug(f"Shapely: {shapely.__version__}")

    # Log CUDA device info
    cuda_dev_num = torch.cuda.device_count()
    logger.debug(f"CUDA devices avail: {cuda_dev_num}")
    for i in range(0, cuda_dev_num):
        logger.debug(
            f"CUDA properties for device {i}: {torch.cuda.get_device_properties(i)}"
        )

    if cuda_dev_num == 0:
        raise ValueError("No GPU devices found. GPU required for inference.")

    logger.info("Starting...")

    if os.name == "nt":
        from multiprocessing import freeze_support

        freeze_support()

    main()


if __name__ == "__main__":
    init()
