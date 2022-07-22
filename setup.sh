#!/bin/sh
git submodule update --init --recursive
ROOT=$PWD

# Install hloc
cd $ROOT/Hierarchical-Localization
pip3 install -e .

# Install pixloc
cd $ROOT/pixloc
pip3 install -e .

# Download pixloc snapshots
PIXLOC_SNAPSHOTS=$ROOT/pixloc/outputs/training
if [ -d "$PIXLOC_SNAPSHOTS" ]
then
	echo "Checkpoints already exist"
else
	python3 -m pixloc.download --select checkpoints
fi

# Install pixsfm
cd $ROOT/pixel-perfect-sfm
pip3 install -e .

# Install instant-ngp
cd $ROOT/instant-ngp
export PATH="/usr/local/cuda-11.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH"
cmake . -B build
cmake --build build --config RelWithDebInfo -j 16
export PATH="$PATH:$PWD/scripts"

cd $ROOT
export PYTHONPATH=$PYTHONPATH:$ROOT
export PROJECT_ROOT=$ROOT
