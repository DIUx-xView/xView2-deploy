import glob
import inference
from raster_processing import *

from tqdm import tqdm


# TODO: Clean up directory structure
# TODO: gather input and output files from folders --> create pre and post mosaic --> create intersection --> get chips from intersection for pre/post --> extract geotransform per chip --> hand off to inference --> georef outputs

PRE_DIR = '/Users/lb/Documents/PycharmProjects/xView2_FDNY/tests/data_small/input/pre'
POST_DIR = '/Users/lb/Documents/PycharmProjects/xView2_FDNY/tests/data_small/input/post'
# TODO: Should we clear this directory first?
STAGING_DIR = '/Users/lb/Documents/PycharmProjects/xView2_FDNY/tests/data_small/input/staging'
OUTPUT_DIR = '/Users/lb/Documents/PycharmProjects/xView2_FDNY/tests/data_small/output'
PRE_IN_CRS = None
POST_IN_CRS = 'EPSG:26915'


def main():

    pre_files = get_files(PRE_DIR)
    post_files = get_files(POST_DIR)
    print("Got files")

    # TODO: Can be removed after chip creation is implemented
    #assert string_len_check(pre_files, post_files)

    pre_reproj = []
    post_reproj = []

    print('Re-projecting...')

    for file in tqdm(pre_files):
        basename = os.path.splitext(os.path.split(file)[1])
        dest_file = os.path.join(STAGING_DIR, 'pre', f'{basename[0]}.tif')

        # Use try to discard images that are not geo images
        # TODO: deconflict this with the function assertions
        try:
            pre_reproj.append(reproject(file, dest_file, PRE_IN_CRS))
        except:
            pass

    for file in tqdm(post_files):
        basename = os.path.splitext(os.path.split(file)[1])
        dest_file = os.path.join(STAGING_DIR, 'post', f'{basename[0]}.tif')

        # Use try to discard images that are not geo images
        try:
            post_reproj.append(reproject(file, dest_file, POST_IN_CRS))
        except:
            pass

    print("Creating pre mosaic")
    pre_mosaic = create_mosaic(pre_reproj, os.path.join(STAGING_DIR, 'mosaics', 'pre.tif'))
    print("Creating post mosaic")
    post_mosaic = create_mosaic(post_reproj, os.path.join(STAGING_DIR, 'mosaics', 'post.tif'))

    extent = get_intersect(pre_mosaic, post_mosaic)

    pre_chips = create_chips(pre_mosaic, os.path.join(OUTPUT_DIR, 'chips', 'pre'), extent)
    post_chips = create_chips(post_mosaic, os.path.join(OUTPUT_DIR, 'chips', 'post'), extent)

    assert (len(pre_chips) == len(post_chips))

    pairs = []
    # TODO: using a for loop seems to only parse about half the values in test data.
    i = 0
    while pre_chips:
        pre = pre_chips.pop(0)
        post = post_chips.pop(0)
        pairs.append(Files(i, pre, post))
        i += 1

    print('Inferring building locations...')
    for obj in tqdm(pairs):
        obj.loc = obj.infer()


class Files(object):

    def __init__(self, ident, pre, post):
        self.ident = ident
        self.pre = os.path.abspath(os.path.join(PRE_DIR, pre))
        self.post = os.path.abspath(os.path.join(POST_DIR, post))
        self.loc = os.path.join(OUTPUT_DIR, 'loc', f'{self.ident}.png')
        self.dmg = os.path.join(OUTPUT_DIR, 'dmg', f'{self.ident}.png')
        self.opts = inference.Options(pre_path=self.pre,
                                      post_path=self.post,
                                      out_loc_path=self.loc,
                                      out_dmg_path=self.dmg
                                      )

    def infer(self):
        """
        Passes object to inference.
        :return: True if successful
        """

        try:
            inference.main(self.opts)
        except Exception as e:
            print(f'Error: {e}')

        return True

    def georef(self):
        # TODO: Name final raster with the extent of the corners
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
