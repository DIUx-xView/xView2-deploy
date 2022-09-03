FROM --platform=linux/amd64 nvidia/cuda:11.0.3-base-ubuntu18.04

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh -O ~/miniconda.sh && \
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

# download first place weights
RUN wget https://xv2-weights.s3.amazonaws.com/first_place_weights.tar.gz && \
    mkdir -p /work/weights && \
    tar -xzvf first_place_weights.tar.gz -C /work/weights && \
    rm first_place_weights.tar.gz

# download backbone weights
RUN wget https://xv2-weights.s3.amazonaws.com/backbone_weights.tar.gz && \
    mkdir --parents /root/.cache/torch/hub/checkpoints/ && \
    tar -xzvf backbone_weights.tar.gz -C /root/.cache/torch/hub/checkpoints/ && \
    rm backbone_weights.tar.gz

# create environment
COPY spec-file.txt /work/locks/
RUN conda create --name xv --file locks/spec-file.txt

RUN conda clean --all --yes

# copy entire directory where docker file is into docker container at /work
# uses . with .dockerignore to ensure folder structure stays correct
COPY zoo/* /work/zoo/

RUN export PATH=/usr/local/cuda/bin:$PATH
RUN export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

COPY utils/* /work/utils/
COPY tests/* /work/tests/
COPY handler.py dataset.py models.py spec-file.txt /work/

VOLUME ["/input/pre", "/input/post", "/input/polys", "/output"]

ENTRYPOINT [ "conda", "run", "-n", "xv", "python", "handler.py","--pre_directory", "/input/pre", "--post_directory", "/input/post", "--output_directory", "/output"]
