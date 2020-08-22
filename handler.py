import glob
import argparse
from pathlib import Path
import sys
import multiprocessing
import resource

import inference
from raster_processing import *
import rasterio.warp

from tqdm import tqdm


# TODO: Clean up directory structure
# TODO: gather input and output files from folders --> create pre and post mosaic --> create intersection --> get chips from intersection for pre/post --> extract geotransform per chip --> hand off to inference --> georef outputs


class Files(object):

    def __init__(self, ident, pre_directory, post_directory, output_directory, pre, post):
        self.ident = ident
        self.pre = pre_directory.joinpath(pre).resolve()
        self.post = post_directory.joinpath(post).resolve()
        self.loc = output_directory.joinpath('loc').joinpath(f'{self.ident}.tif').resolve()
        self.dmg = output_directory.joinpath('dmg').joinpath(f'{self.ident}.tif').resolve()
        self.over = output_directory.joinpath('over').joinpath(f'{self.ident}.tif').resolve()
        self.profile = self.get_profile()
        self.transform = self.profile["transform"]
        self.opts = inference.Options(pre_path=self.pre,
                                      post_path=self.post,
                                      out_loc_path=self.loc,
                                      out_dmg_path=self.dmg,
                                      out_overlay_path=self.over,
                                      geo_profile=self.profile,
                                      vis=True
                                      )

    def get_profile(self):
        with rasterio.open(self.pre) as src:
            return src.profile

    def infer(self):
        """
        Passes object to inference.
        :return: True if successful
        """

        try:
            inference.main(self.opts)
            self.georef(self.loc, 'loc')
            self.georef(self.dmg, 'dmg')
        except Exception as e:
            raise e

        return True

    def georef(self, in_file, path):
        # TODO: Name final raster with the extent of the corners
        pass


def make_staging_structure(staging_path):
    """
    Creates directory structure for staging.
    :param staging_path: Staging path
    :return: True if successful
    """
    Path(f"{staging_path}/pre").mkdir(parents=True, exist_ok=True)
    Path(f"{staging_path}/post").mkdir(parents=True, exist_ok=True)
    Path(f"{staging_path}/mosaics").mkdir(parents=True, exist_ok=True)

    return True


def make_output_structure(output_path):

    """
    Creates directory structure for outputs.
    :param output_path: Output path
    :return: True if succussful
    """

    Path(f"{output_path}/chips/pre").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/chips/post").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/loc").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/dmg").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/over").mkdir(parents=True, exist_ok=True)

    return True


def get_files(dirname, extensions=['.png', '.tif', '.jpg']):

    """
    Gathers list of files for processing from path recursively.
    :param dirname: path to parse
    :param extensions: extensions to match
    :return: list of files matching extensions
    """
    dir_path = Path(dirname)
    files = dir_path.glob('**/*')
    files = [path.resolve() for path in files]

    match = [f for f in files if f.suffix in extensions]
    return match


def reproject_helper(args, raster_tuple, procnum, return_dict):
    (pre_post, src_crs, raster_file) = raster_tuple
    basename = raster_file.stem
    dest_file = args.staging_directory.joinpath('pre').joinpath(f'{basename}.tif')
    try:
        return_dict[procnum] = (pre_post, reproject(raster_file, dest_file, src_crs, args.destination_crs))
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description='Create arguments for xView 2 handler.')

    parser.add_argument('--pre_directory', metavar='/path/to/pre/files/', type=Path)
    parser.add_argument('--post_directory', metavar='/path/to/post/files/', type=Path)
    parser.add_argument('--staging_directory', metavar='/path/to/staging/', type=Path)
    parser.add_argument('--output_directory', metavar='/path/to/output/', type=Path)
    parser.add_argument('--pre_crs', help='The Coordinate Reference System (CRS) for the pre-disaster imagery.')
    parser.add_argument('--post_crs', help='The Coordinate Reference System (CRS) for the post-disaster imagery.')
    parser.add_argument('--destination_crs', default='EPSG:4326', help='The Coordinate Reference System (CRS) for the output overlays.')
    parser.add_argument('--create_overlay_mosaic', default=False, action='store_true', help='True/False to create a mosaic out of the overlays')

    args = parser.parse_args()

    make_staging_structure(args.staging_directory)
    make_output_structure(args.output_directory)

    pre_files = get_files(args.pre_directory)
    post_files = get_files(args.post_directory)

    print('Re-projecting...')

    # Run reprojection in parallel processes
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []

    # Some data hacking to make it more efficient for multiprocessing
    pre_files = [("pre", args.pre_crs, x) for x in pre_files]
    post_files = [("post", args.post_crs, x) for x in post_files]
    files = pre_files + post_files

    # Launch multiprocessing jobs
    for idx, f in enumerate(files):
        p = multiprocessing.Process(target=reproject_helper, args=(args, f, idx, return_dict))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()

    reproj = [x for x in return_dict.values() if x[1] is not None]
    pre_reproj = [x[1] for x in reproj if x[0] == "pre"]
    post_reproj = [x[1] for x in reproj if x[0] == "post"]

    print("Creating pre mosaic")
    pre_mosaic = create_mosaic(pre_reproj, Path(f"{args.staging_directory}/mosaics/pre.tif"))
    print("Creating post mosaic")
    post_mosaic = create_mosaic(post_reproj, Path(f"{args.staging_directory}/mosaics/post.tif"))

    extent = get_intersect(pre_mosaic, post_mosaic)

    pre_chips = create_chips(pre_mosaic, args.output_directory.joinpath('chips').joinpath('pre'), extent)
    post_chips = create_chips(post_mosaic, args.output_directory.joinpath('chips').joinpath('post'), extent)

    assert len(pre_chips) == len(post_chips)

    pairs = []
    for idx, (pre, post) in enumerate(zip(pre_chips, post_chips)):
        pairs.append(Files(
            pre.stem,
            args.pre_directory,
            args.post_directory,
            args.output_directory,
            pre,
            post)
            )

    print('Inferring building locations...')
    for obj in tqdm(pairs):
        obj.loc = obj.infer()

    if args.create_overlay_mosaic:
        print("Creating overlay mosaic")
        p = Path(args.output_directory) / "over"
        overlay_files = p.glob('*')
        overlay_files = [x for x in overlay_files]

        # This is some hacky, dumb shit
        # There is a limit on how many file descriptors we can have open at once
        # So we will up that limit for a bit and then set it back
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if len(overlay_files) >= soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (len(overlay_files) + 10, hard))

        overlay_mosaic = create_mosaic(overlay_files, Path(f"{args.staging_directory}/mosaics/overlay.tif"))

        # Reset soft limit
        if len(overlay_files) + 10 < soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


if __name__ == '__main__':
    main()
