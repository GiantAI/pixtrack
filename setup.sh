#!/bin/sh
git submodule update --init --recursive
ROOT=$PWD
cd $ROOT/Hierarchical-Localization
echo $PWD
pip3 install -e .
cd $ROOT/pixel-perfect-sfm
echo $PWD
pip3 install -r requirements.txt
pip3 install -e .
cd $ROOT
