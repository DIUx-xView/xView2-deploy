FROM --platform=linux/amd64 ubuntu:18.04

# Use a fixed apt-get repo to stop intermittent failures due to flaky httpredir connections,
# as described by Lionel Chan at http://stackoverflow.com/a/37426929/5881346
RUN sed -i "s/httpredir.debian.org/debian.uchicago.edu/" /etc/apt/sources.list && \
    apt-get update && apt-get install -y build-essential
    
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion zip unzip
    
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

ENV PATH /opt/conda/bin:$PATH

SHELL ["/bin/bash", "-c"]

WORKDIR /work

# copy entire directory where docker file is into docker container at /work
# uses . with .dockerignore to ensure folder structure stays correct
COPY . /work/

RUN conda install --file conda-linux-64.lock

VOLUME ["/input/pre", "/input/post", "/input/polys", "/output"]

ENTRYPOINT [ "python", "test.py" ]

# ENTRYPOINT ["python", "handler.py", \
#     "--pre_directory", "/input/pre", \
#     "--post_directory", "/input/post",\
#     "--output_directory", "/output"\
#     ]

# Debug log:
#  1: unable to call Conda, numerous missing libraries