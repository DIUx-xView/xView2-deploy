import glob
import inference
from raster_processing import *
# TODO: Clean up directory structure
# TODO: gather input and output files from folders --> create pre and post mosaic --> create intersection --> get chips from intersection for pre/post --> extract geotransform per chip --> hand off to inference --> georef outputs

PRE_DIR = 'tests/data/input/pre'
POST_DIR = 'tests/data/input/post'
# TODO: Should we clear this directory first?
STAGING_DIR = 'tests/data/input/staging'
OUTPUT_DIR = 'tests/data/output'


def main():

    pre_files = get_files(PRE_DIR)
    post_files = get_files(POST_DIR)

    # TODO: Can be removed after chip creation is implemented
    #string_len_check(pre_files, post_files)

    pre_reproj = []
    post_reproj = []

    for file in pre_files:
        basename = os.path.splitext(os.path.split(file)[1])
        dest_file = os.path.join(STAGING_DIR, 'pre', f'{basename[0]}.tif')

        # Use try to discard images that are not geo images
        try:
            pre_reproj.append(reproject(file, dest_file))
        except:
            pass

    for file in post_files:
        basename = os.path.splitext(os.path.split(file)[1])
        dest_file = os.path.join(STAGING_DIR, 'post', f'{basename[0]}.tif')
        post_reproj.append(reproject(file, dest_file))

        # Use try to discard images that are not geo images
        try:
            pre_reproj.append(reproject(file, dest_file))
        except:
            pass

    pre_mosaic = create_mosaic(pre_reproj, os.path.join(STAGING_DIR, 'mosaics', 'pre.tif'))
    post_mosaic = create_mosaic(post_reproj, os.path.join(STAGING_DIR, 'mosaics', 'post.tif'))
    extent = get_intersect(pre_mosaic, post_mosaic)
    print(extent)
    create_chips(pre_mosaic, '/Users/lb/Documents/PycharmProjects/xView2_FDNY/tests/data/output/chips/pre')
    create_chips(post_mosaic, '/Users/lb/Documents/PycharmProjects/xView2_FDNY/tests/data/output/chips/post')

    # TODO: Create our package object


class Files(object):

    def __init__(self, pre, post):
        self.pre = os.path.abspath(os.path.join(PRE_DIR, pre))
        self.post = os.path.abspath(os.path.join(POST_DIR, post))
        self.base_num = self.check_extent()
        self.output_loc = os.path.abspath(os.path.join(OUTPUT_DIR, 'loc', f'{self.base_num}_loc.png'))
        self.output_dmg = os.path.abspath(os.path.join(OUTPUT_DIR, 'dmg', f'{self.base_num}_dmg.png'))
        self.opts = inference.Options(self.pre, self.post, self.output_loc, self.output_dmg)

    def check_extent(self):
        """
        Check that our pre and post are the same extent
        Note:
        Currently only checks that the number sequence matches for both the pre and post images.
        :return: True if numbers match
        """
        pre_base = ''.join([digit for digit in self.pre if digit.isdigit()])
        post_base = ''.join([digit for digit in self.post if digit.isdigit()])
        if pre_base == post_base:
            return pre_base

    def infer(self):
        """
        Passes object to inference.
        :return: True if successful
        """

        try:
            # TODO: Not sure why opts seems to be a list.
            inference.main(self.opts[0])
        except Exception as e:
            print(f'File: {self.pre}. Error: {e}')
            return False

        return True

    def georef(self):
        pass


def make_output_structure(path):
    pass


def get_files(dirname, extensions=['.png', '.tif', '.jpg'], recursive=True):
    files = glob.glob(f'{dirname}/**', recursive=recursive)
    match = [os.path.abspath(file) for file in files if os.path.splitext(file)[1].lower() in extensions]
    return match


def string_len_check(pre, post):

    if len(pre) != len(post):
        # TODO: Add some helpful info on why this failed
        return False

    return True


if __name__ == '__main__':
    main()
