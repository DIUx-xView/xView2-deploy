import subprocess
from sys import exit

res = [0.3, 0.6, 0.9]
cuda = 'CUDA_VISIBLE_DEVICES=0,1'
pre_dir = 'pre_dir'
post_dir = 'post_dir'
out_dir = 'out_dir'
stage_dir = 'stage_dir'
n_procs = 64
batch_size = 4
workers = 6


for r in res:
    command = f'{cuda} python ' \
              f'handler.py ' \
              f'--pre_directory {pre_dir} ' \
              f'--post_directory {post_dir} ' \
              f'--output_directory {out_dir} ' \
              f'--staging_directory {stage_dir} ' \
              f'--destination_crs EPSG:4326 ' \
              f'--post_crs EPSG:26915 ' \
              f'--output_resolution {r} ' \
              f'--model_weight_path weights/weight.pth ' \
              f'--model_config_path configs/model.yaml ' \
              f'--n_procs {n_procs} '\
              f'--batch_size {batch_size} ' \
              f'--num_workers {workers} ' \
              f'--dp_mode ' \
              f'--create_shapefiles'\
        .split()
    raise Exception('Remove me, Dummy')
    subprocess.call(command, stdout=True)