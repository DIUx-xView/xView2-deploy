import argparse
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm

from raster_processing import *


def main(args):
    dmg_files = list(Path(args.output_directory / 'dmg_noref').glob('*'))
    loc_files = list(Path(args.output_directory / 'loc_noref').glob('*'))

    pre_files = [p.parents[1] / 'chips' / 'post' / (p.stem + '_post.tif') for p in dmg_files]

    for prepath, dmgpath, locpath in tqdm(zip(pre_files, dmg_files, loc_files), total=len(dmg_files)):
        preds = rasterio.open(prepath)
        meta = preds.meta.copy()
        meta.update(count=1)

        dmgds = rasterio.open(dmgpath).read()
        locds = rasterio.open(locpath).read()

        outpath = args.output_directory / 'dmg' / dmgpath.name
        with rasterio.open(outpath, 'w', **meta) as outds:
            outds.write(dmgds)

        outpath = args.output_directory / 'loc' / dmgpath.name
        with rasterio.open(outpath, 'w', **meta) as outds:
            outds.write(locds)

        dmgds = np.squeeze(rasterio.open(dmgpath).read().transpose(1,2,0))
        meta.update(count=3)

        if args.create_overlay_mosaic:
            cls = dmgds
            mask_map_img = np.zeros((cls.shape[0], cls.shape[1], 3), dtype=np.uint8)
            mask_map_img[cls == 1] = (255, 255, 255)
            mask_map_img[cls == 2] = (229, 255, 50)
            mask_map_img[cls == 3] = (255, 159, 0)
            mask_map_img[cls == 4] = (255, 0, 0)

            out_file = args.output_directory / 'over' / dmgpath.name
            with rasterio.open(out_file, 'w', **meta) as dst:
                # Go from (x, y, bands) to (bands, x, y)
                mask_map_img = np.flipud(mask_map_img)
                mask_map_img = np.rot90(mask_map_img, 3)
                mask_map_img = np.moveaxis(mask_map_img, [0, 1, 2], [2, 1, 0])
                dst.write(mask_map_img)


    if args.create_overlay_mosaic:
        print("Creating overlay mosaic")
        p = Path(args.output_directory) / "over"
        overlay_files = p.glob('*')
        overlay_files = [x for x in overlay_files]
        overlay_mosaic = create_mosaic(overlay_files, Path(f"{args.output_directory}/mosaics/overlay.tif"))

    if args.create_shapefile:
        print('Creating shapefile')
        create_shapefile(Path(args.output_directory) / 'dmg',
                         Path(args.output_directory).joinpath('shapes') / 'damage.shp',
                         args.destination_crs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create arguments for xView 2 handler.')

    parser.add_argument('--staging_directory', metavar='/path/to/staging/', type=Path)
    parser.add_argument('--output_directory', metavar='/path/to/output/', type=Path, required=True)
    parser.add_argument('--is_use_gpu', action='store_true', help="If True, use GPUs")
    parser.add_argument('--n_procs', default=4, help="Number of processors for multiprocessing", type=int)
    parser.add_argument('--batch_size', default=16, help="Number of chips to run inference on at once", type=int)
    parser.add_argument('--num_workers', default=8, help="Number of workers loading data into RAM. Recommend 4 * num_gpu", type=int)
    parser.add_argument('--pre_crs', help='The Coordinate Reference System (CRS) for the pre-disaster imagery.')
    parser.add_argument('--post_crs', help='The Coordinate Reference System (CRS) for the post-disaster imagery.')
    parser.add_argument('--destination_crs', default='EPSG:4326', help='The Coordinate Reference System (CRS) for the output overlays.')
    parser.add_argument('--create_overlay_mosaic', default=False, action='store_true', help='True/False to create a mosaic out of the overlays')
    parser.add_argument('--create_shapefile', default=False, action='store_true', help='True/False to create shapefile from damage overlay')

    args = parser.parse_args()

    main(args)
