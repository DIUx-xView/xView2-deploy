import glob
import argparse
from pathlib import Path

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
        self.loc = output_directory.joinpath('loc').joinpath(f'{self.ident}.png').resolve()
        self.dmg = output_directory.joinpath('dmg').joinpath(f'{self.ident}.png').resolve()
        self.opts = inference.Options(pre_path=self.pre,
                                      post_path=self.post,
                                      out_loc_path=self.loc,
                                      out_dmg_path=self.dmg
                                      )
        self.profile = self.get_profile()
        self.transform = self.profile["transform"]

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


def main():
    parser = argparse.ArgumentParser(description='Create arguments for xView 2 handler.')

    parser.add_argument('--pre_directory', metavar='/path/to/pre/files/', type=Path)
    parser.add_argument('--post_directory', metavar='/path/to/post/files/', type=Path)
    parser.add_argument('--staging_directory', metavar='/path/to/staging/', type=Path)
    parser.add_argument('--output_directory', metavar='/path/to/output/', type=Path)
    parser.add_argument('--pre_crs', help='The Coordinate Reference System (CRS) for the pre-disaster imagery.')
    parser.add_argument('--post_crs', help='The Coordinate Reference System (CRS) for the post-disaster imagery.')
    parser.add_argument('--destination_crs', metavar='EPSG:4326', help='The Coordinate Reference System (CRS) for the output overlays.')

    args = parser.parse_args()

    make_staging_structure(args.staging_directory)
    make_output_structure(args.output_directory)

    pre_files = get_files(args.pre_directory)
    post_files = get_files(args.post_directory)
    print("Got files")

    # TODO: Can be removed after chip creation is implemented
    #assert string_len_check(pre_files, post_files)

    pre_reproj = []
    post_reproj = []

    print('Re-projecting...')

    for file in tqdm(pre_files):
        basename = file.stem
        dest_file = args.staging_directory.joinpath('pre').joinpath(f'{basename}.tif')

        # Use try to discard images that are not geo images
        # TODO: deconflict this with the function assertions
        try:
            pre_reproj.append(reproject(file, dest_file, args.pre_crs, args.destination_crs))
        except ValueError:
            pass

    for file in tqdm(post_files):
        basename = file.stem
        dest_file = args.staging_directory.joinpath('post').joinpath(f'{basename}.tif')

        # Use try to discard images that are not geo images
        try:
            post_reproj.append(reproject(file, dest_file, args.post_crs, args.destination_crs))
        except ValueError:
            pass

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
            idx,
            args.pre_directory,
            args.post_directory,
            args.output_directory,
            pre,
            post)
            )

    print('Inferring building locations...')
    for obj in tqdm(pairs):
        obj.loc = obj.infer()


if __name__ == '__main__':
    main()
