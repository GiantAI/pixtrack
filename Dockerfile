FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
 
MAINTAINER Prajwal Chidananda prajwal.chidananda@giant.ai

ENV DEBIAN_FRONTEND=noninteractive 

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
  apt-utils \
  build-essential \
  sudo \
  curl \
  gdb \
  git \
  pkg-config \
  python-numpy \
  python-dev \
  python-setuptools \
  python3-pip \
  python3-opencv \
  python3-dev \
  rsync \
  wget \
  vim \
  unzip \
  zip \
  htop \
  libboost-program-options-dev \
  libboost-filesystem-dev \
  libboost-graph-dev \
  libboost-regex-dev \
  libboost-system-dev \
  libboost-test-dev \
  libsuitesparse-dev \
  libfreeimage-dev \
  libgoogle-glog-dev \
  libgflags-dev \
  libglew-dev \
  qtbase5-dev \
  libqt5opengl5-dev \
  libcgal-dev \
  libcgal-qt5-dev \
  libfreetype6-dev \
  libpng-dev \
  libzmq3-dev \
  ffmpeg \
  software-properties-common \
  libatlas-base-dev \
  libsuitesparse-dev \
  libgoogle-glog-dev \
  libsuitesparse-dev \
  libmetis-dev \
  libglfw3-dev \
  imagemagick \
  screen \
  liboctomap-dev \
  libfcl-dev \
  libhdf5-dev \
  libopenexr-dev \
  libxi-dev \
  libomp-dev \
  libxinerama-dev \
  libxcursor-dev \
  libxrandr-dev \
  && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* && \
  apt-get clean && rm -rf /tmp/* /var/tmp/*

# CMake
RUN pip3 install --upgrade cmake

# Eigen
WORKDIR /opt
RUN git clone --depth 1 --branch 3.4.0 https://gitlab.com/libeigen/eigen.git
RUN cd eigen && mkdir build && cd build && cmake .. && make install

# Ceres solver
WORKDIR /opt
RUN apt-get update
RUN git clone https://ceres-solver.googlesource.com/ceres-solver
WORKDIR /opt/ceres-solver
RUN git checkout 2.1.0rc2
RUN mkdir build
WORKDIR /opt/ceres-solver/build
RUN cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
RUN make -j
RUN make install

# Colmap
WORKDIR /opt
RUN git clone https://github.com/colmap/colmap
WORKDIR /opt/colmap
RUN git checkout dev
RUN mkdir build
WORKDIR /opt/colmap/build
RUN cmake ..
RUN make -j
RUN make install

# PyRender
WORKDIR /
RUN apt update
RUN wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb
RUN dpkg -i ./mesa_18.3.3-0.deb || true
RUN apt install -y -f
RUN git clone https://github.com/mmatl/pyopengl.git
RUN pip3 install ./pyopengl
RUN pip3 install pyrender

RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install imageio
RUN pip3 install imageio-ffmpeg
RUN pip3 install matplotlib
RUN pip3 install configargparse
RUN pip3 install tensorboard
RUN pip3 install tqdm
RUN pip3 install opencv-python
RUN pip3 install ipython
RUN pip3 install sklearn
RUN pip3 install pandas
RUN pip3 install dash
RUN pip3 install jupyter-dash
RUN pip3 install Pillow
RUN pip3 install scipy
RUN pip3 install scikit-image 
RUN pip3 install tensorflow
RUN pip3 install pytorch-lightning
RUN pip3 install test-tube
RUN pip3 install kornia==0.2.0
RUN pip3 install PyMCubes
RUN pip3 install pycollada
RUN pip3 install trimesh
RUN pip3 install pyglet
RUN pip3 install plyfile
RUN pip3 install open3d
RUN pip3 install scikit-video
RUN pip3 install cmapy
RUN pip3 install scikit-image==0.16.2
RUN pip3 install jupyter_http_over_ws
RUN pip3 install plotly
RUN pip3 install python-fcl
RUN pip3 install opencv-contrib-python
RUN pip3 install prettytable
RUN pip3 install yacs
RUN pip3 install torchfile
RUN pip3 install munkres
RUN pip3 install chumpy
RUN pip3 install shyaml
RUN pip3 install PyYAML>=5.1.2
RUN pip3 install numpy-quaternion
RUN pip3 install pygame
RUN pip3 install keyboard
RUN pip3 install transforms3d
RUN pip3 install bvhtoolbox
RUN pip3 install vedo
RUN pip3 install imgaug
RUN pip3 install lap
RUN pip3 install smplx
RUN pip3 install pycocotools
RUN pip3 install ipdb 
RUN pip3 install lpips 
RUN pip3 install jax==0.2.9
RUN pip3 install jaxlib>=0.1.57
RUN pip3 install flax>=0.3.1
RUN pip3 install pyyaml 
RUN pip3 install pymcubes
RUN pip3 install --upgrade jax[cuda11_cudnn805] -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN pip3 install rtree
RUN pip3 install --upgrade git+https://github.com/colmap/pycolmap
RUN pip3 install h5py
RUN pip3 install omegaconf
RUN pip3 install packaging
ENV FORCE_CUDA=1
RUN pip3 install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
RUN pip3 install svox
#RUN echo "alias python=python3" >> .bashrc
