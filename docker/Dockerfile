FROM nvidia/cuda:10.2-cudnn7-devel

ENV DEBIAN_FRONTEND noninteractive
ENV LC_ALL="C.UTF-8"
ENV LANG="C.UTF-8"

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    ca-certificates \
    cmake \
    git \
    python3-dev \
    python3-opencv \
    ninja-build \
    sudo \
    vim \
    wget \
 && rm -rf /var/lib/apt/lists/*
RUN ln -sv /usr/bin/python3 /usr/bin/python

# Setup paths
RUN mkdir /workspace
RUN mkdir /server

# Create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo && chown -R appuser /workspace
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

# Install pip
ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py --user && \
    pip install --upgrade pip && \
    rm get-pip.py
RUN /usr/bin/python3 -m pip install --upgrade pip

# install py packages
RUN pip install torch==1.7.0 torchvision==0.8.1 -f https://download.pytorch.org/whl/cu102/torch_stable.html
RUN pip install nvidia-pyindex==1.0.6
RUN pip install \
    requests \
    Flask>=1.1.1 \
    flask-restplus>=0.13.0 \
    opencv-python-headless \
    Pillow>=7.1 \
    alphashape==1.1.0 \
    waitress==1.4.4 \
    fastapi==0.54.1 \
    uvicorn==0.11.5 \
    werkzeug==0.16.1 \
    shapely==1.7.1 \
    paste==3.5.0 \
    click>=7.0 \
    matplotlib \
    python-magic \
    tritonclient[all]

# Change workdir
WORKDIR /workspace
