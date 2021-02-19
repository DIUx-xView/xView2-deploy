import subprocess
from pathlib import Path

res = ['3e-6', '6e-6', '9e-6']
cuda = 'CUDA_VISIBLE_DEVICES=0,1,2,3'
pre_dir = Path('~/data/joplin tornado/input/pre')
post_dir = Path('~/data/joplin tornado/input/post')
out_dir = Path('~/data/dyn_res')
stage_dir = Path('~data/dyn_res/')
n_procs = 64
batch_size = 4
workers = 16


for r in res:
    out_dir_res = out_dir / r / 'output'
    stage_dir_res = stage_dir / r / 'staging'
    command = f'{cuda} python ' \
              f'handler.py ' \
              f'--pre_directory {pre_dir} ' \
              f'--post_directory {post_dir} ' \
              f'--output_directory {out_dir_res} ' \
              f'--staging_directory {stage_dir_res} ' \
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
    #raise Exception('Remove me, Dummy')
    subprocess.call(command, stdout=True)