# System setup

xView2 inference requires a tremendous amount of computing power. Currently, CPU inference is wildly
impractical. To that end, unless you have a dedicated workstation with ample GPU power such as an Nvidia DGX station,
we recommend a cloud based solution such as AWS or Google Cloud Compute utilizing a GPU optimized instance. Prices vary
on instance type and area to be inferred. Example instances:

1. AWS EC2
   1. P4d.24xlarge
   2. P3.16xlarge
2. G Cloud
   1. Todo!

# Installation

## Install from source

**Note**: Only tested on Linux systems.

1. Close repository: `https://github.com/DIUx-xView/xView2-deploy`.
2. Create Conda environment: `conda create --name xv2 --file spec-file.txt`.
3. Activate conda environment: `conda activate xv2`.

## Docker

Build docker image with:
`docker build -f "Dockerfile" -t xview2infer:latest "."`
**Note:** set --shm-size to something appropriate for your machine.
and run with:
`docker run --rm --gpus=all --shm-size="128g" -v /path/to/pre:/input/pre -v /path/to/post:/input/post -v /path/to/output:/output xview2infer:latest`
if utilizing input polygons, mount volume containing polygons and pass file as argument. ie:
`docker run --rm --gpus=all --shm-size="128g" -v /path/to/pre:/input/pre -v /path/to/post:/input/post -v /path/to/output:/output -v /path/to/polygons:/input/polys xview2infer:latest --bldg_polys /input/polys/polys.geojson`
**Note:** full filename must be passed to --bldg_polys argument

# Usage

| Argument             | Required | Default   | Help                                                                                                                                                                            |
|----------------------|----------|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --pre_directory      | Yes      | None      | Directory containing pre-disaster imagery. This is searched recursively.                                                                                                        |
| --post_directory     | Yes      | None      | Directory containing post-disaster imagery. This is searched recursively.                                                                                                       |
| --output_directory   | Yes      | None      | Directory to store output files. This will be created if it does not exist. Existing files may be overwritten.                                                                  |
| --batch_size         | Yes      | 2         | Number of chips to run inference on at once                                                                                                                                     |
| --pre_crs            | No       | None      | The Coordinate Reference System (CRS) for the pre-disaster imagery. This will only be utilized if images lack CRS data.                                                         |
| --post_crs           | No       | None      | The Coordinate Reference System (CRS) for the post-disaster imagery. This will only be utilized if images lack CRS data.                                                        |
| --destination_crs    | No       | EPSG:4326 | The Coordinate Reference System (CRS) for the output overlays.                                                                                                                  |
| --output_resolution  | No       | None      | Override minimum resolution calculator. This should be a lower resolution (higher number) than source imagery for decreased inference time. Must be in units of destinationCRS. |
| --dp_mode            | No       | False     | Run models serially, but using DataParallel                                                                                                                                     |
| --save_intermediates | No       | False     | Store intermediate runfiles                                                                                                                                                     |
| --aoi_file           | No       | None      | Shapefile or GeoJSON file of AOI polygons                                                                                                                                       |
| --bldg_polys         | No       | None      | Shapefile or GeoJSON file of input building footprints                                                                                                                          |

# Example invocation for damage assessment

On 2 GPUs:
`CUDA_VISIBLE_DEVICES=0,1 python handler.py --pre_directory <pre dir> --post_directory <post dir> --output_directory <output dir> --aoi_file <aoi file (GeoJSON or shapefile)>`

# Notes:

- CRS between input types (pre/post/building footprints/AOI) need not match. However CRS _within_ input types must match.

# Sources

**xView2 1st place solution**

- Model weights from 1st place solution for "xView2: Assess Building Damage" challenge. https://github.com/DIUx-xView/xView2_first_place
- More information from original submission see commit: 3fe4a7327f1a19b8c516e0b0930c38c29ac3662b
