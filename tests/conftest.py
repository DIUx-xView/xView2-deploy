from pathlib import Path
import pytest


class Args:

    def __init__(self,
                 staging_path=None,
                 output_path=None,
                 pre_directory='tests/data/input/pre',
                 post_directory='tests/data/input/post',
                 n_procs=4,
                 batch_size=1,
                 num_workers=8,
                 pre_crs=None,
                 post_crs=None,
                 destination_crs=None,
                 output_resolution=None,
                 save_intermediates=False,
                 agol_user='',
                 agol_password='',
                 agol_feature_service='',
                 dp_mode=True
                 ):

        self.output_directory = output_path
        self.staging_directory = staging_path
        self.staging_directory = staging_path
        self.output_directory = output_path
        self.pre_directory = Path(pre_directory)
        self.post_directory = Path(post_directory)
        self.n_procs = n_procs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pre_crs = pre_crs
        self.post_crs = post_crs
        self.destination_crs = destination_crs
        self.output_resolution = output_resolution
        self.save_intermediates = save_intermediates
        self.agol_user = agol_user
        self.agol_password = agol_password
        self.agol_feature_service = agol_feature_service
        self.dp_mode = dp_mode


#### Dataframe fixtures ####

@pytest.fixture
def df_1_int_with_df_2_and_3():
    pass


@pytest.fixture
def df2_int_with_df_1():
    pass


@pytest.fixture()
def df3_int_with_df_1():
    pass