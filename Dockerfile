FROM --platform=linux/amd64 nvidia/cuda:11.4.0-base-ubuntu18.04

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
ENV LD_LIBRARY_PATH /usr/local/cuda-11.4/lib64:/usr/local/cuda-11.4/extras/CUPTI/lib64:$LD_LIBRARY_PATH

SHELL ["/bin/bash", "-c"]

WORKDIR /work

# copy entire directory where docker file is into docker container at /work
# uses . with .dockerignore to ensure folder structure stays correct
COPY weights/* /work/weights/
COPY zoo/* /work/zoo/

# to export spec-file run: 'conda list --explicit > spec-file.txt'
# BUG: attempting to install spec file to base environment breaks conda
COPY spec-file.txt /work/locks/
RUN conda create --name xv --file locks/spec-file.txt

RUN conda activate xv

COPY utils/* /work/utils/
COPY tests/* /work/tests/
COPY test.py handler.py dataset.py models.py spec-file.txt /work/

VOLUME ["/input/pre", "/input/post", "/input/polys", "/output"]

ENTRYPOINT [ "python", "test.py" ]
#ENTRYPOINT [ "python", "handler.py", "--pre_directory", "/input/pre", "--post_directory", "/input/post", "--output_directory", "/output", "--n_procs", "8", "--batch_size", "2", "--num_workers", "4" ]
