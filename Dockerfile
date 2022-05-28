FROM --platform=linux/amd64 nvidia/cuda:11.0-base-ubuntu18.04

# to fix key rotation issue
RUN rm /etc/apt/sources.list.d/cuda.list && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

SHELL ["/bin/bash", "-c"]

WORKDIR /work

# copy entire directory where docker file is into docker container at /work
# uses . with .dockerignore to ensure folder structure stays correct
COPY weights/* /work/weights/
COPY zoo/* /work/zoo/

# download backbone weights
# TODO: Move this to S3 bucket: S3 bucket name: xv2-weights
RUN mkdir --parents /root/.cache/torch/hub/checkpoints/
RUN wget --quiet https://download.pytorch.org/models/resnet34-b627a593.pth -O /root/.cache/torch/hub/checkpoints/resnet34-b627a593.pth
RUN wget --quiet http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth -O /root/.cache/torch/hub/checkpoints/se_resnext50_32x4d-a260b3a4.pth
RUN wget --quiet http://data.lip6.fr/cadene/pretrainedmodels/dpn92_extra-b040e4a9b.pth -O /root/.cache/torch/hub/checkpoints/dpn92_extra-b040e4a9b.pth
RUN wget --quiet http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth -O /root/.cache/torch/hub/checkpoints/senet154-c7b49a05.pth

# to export spec-file run: 'conda list --explicit > spec-file.txt'
# BUG: attempting to install spec file to base environment breaks conda
COPY spec-file.txt /work/locks/
RUN conda create --name xv --file locks/spec-file.txt

RUN export PATH=/usr/local/cuda/bin:$PATH
RUN export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

COPY utils/* /work/utils/
COPY tests/* /work/tests/
COPY handler.py dataset.py models.py spec-file.txt /work/

VOLUME ["/input/pre", "/input/post", "/input/polys", "/output"]

ENTRYPOINT [ "conda", "run", "-n", "xv", "python", "handler.py","--pre_directory", "/input/pre", "--post_directory", "/input/post", "--output_directory", "/output", "--n_procs", "8", "--batch_size", "2", "--num_workers", "4", "--dp_mode" ]
