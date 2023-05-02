FROM --platform=linux/amd64 continuumio/miniconda3 AS build

# create environment
COPY spec-file.txt /work/locks/
RUN conda create --name xv --file /work/locks/spec-file.txt

RUN conda install -c conda-forge conda-pack

# Use conda-pack to create a standalone enviornment in /venv:
RUN conda-pack -n xv -o env.tar
RUN mkdir /venv
RUN tar -xf env.tar -C /venv
RUN rm env.tar

# We've put venv in same path it'll be in final image,
# so now fix up paths:
RUN /venv/bin/conda-unpack


FROM --platform=linux/amd64 nvidia/cuda:11.0.3-base-ubuntu18.04 as runtime

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy /venv from the previous stage:
COPY --from=build /venv /venv

# 'hactivate' our venv
ENV VIRTUAL_ENV=/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# to fix proj path issue
ENV PROJ_LIB=/venv/share/proj/

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

# copy entire directory where docker file is into docker container at /work
# uses . with .dockerignore to ensure folder structure stays correct
COPY zoo/* /work/zoo/

RUN export PATH=/usr/local/cuda/bin:$PATH
RUN export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

COPY utils/* /work/utils/
COPY tests/* /work/tests/
COPY handler.py dataset.py models.py spec-file.txt /work/

VOLUME ["/input/pre", "/input/post", "/input/polys", "/output"]

ENTRYPOINT [ "python", "handler.py","--pre_directory", "/input/pre", "--post_directory", "/input/post", "--output_directory", "/output"]
