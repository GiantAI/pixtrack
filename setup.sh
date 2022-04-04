#!/bin/sh
git submodule update --init --recursive
ROOT=$PWD

# Install hloc
cd $ROOT/Hierarchical-Localization
pip3 install -e .

# Install pixloc
cd $ROOT/pixloc
pip3 install -e .
python3 -m pixloc.download --select checkpoints

# Install pixsfm
cd $ROOT/pixel-perfect-sfm
pip3 install -r requirements.txt
pip3 install -e .

cd $ROOT
export PYTHONPATH=$PYTHONPATH:$ROOT
