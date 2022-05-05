FROM --platform=linux/amd64 continuumio/miniconda3:latest

# Use a fixed apt-get repo to stop intermittent failures due to flaky httpredir connections,
# as described by Lionel Chan at http://stackoverflow.com/a/37426929/5881346
RUN sed -i "s/httpredir.debian.org/debian.uchicago.edu/" /etc/apt/sources.list && \
    apt-get update && apt-get install -y build-essential
    
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion zip unzip
    
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV PATH /opt/conda/bin:$PATH

SHELL ["/bin/bash", "-c"]

WORKDIR /work

# copy entire directory where docker file is into docker container at /work
# uses . with .dockerignore to ensure folder structure stays correct
COPY weights/* /work/weights/
COPY zoo/* /work/zoo/

COPY spec-file.txt /work/locks/
# to export spec-file run: 'conda list --explicit > spec-file.txt'
RUN conda install --file locks/spec-file.txt

COPY . /work/

VOLUME ["/input/pre", "/input/post", "/input/polys", "/output"]

ENTRYPOINT [ "python", "handler.py", "--pre_directory", "/input/pre", "--post_directory", "/input/post", "--output_directory", "/output" ]

# ENTRYPOINT ["python", "handler.py", \
#     "--pre_directory", "/input/pre", \
#     "--post_directory", "/input/post",\
#     "--output_directory", "/output"\
#     ]
