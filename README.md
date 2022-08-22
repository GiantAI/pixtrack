# PixTrack

A Computer Vision method for Object Tracking which uses NeRF templates and feature-metric alignment to robustly track the 6DoF pose of a known object.

[![PixTrack](https://img.youtube.com/vi/yQtCUS3i9nc/0.jpg)](https://www.youtube.com/watch?v=yQtCUS3i9nc)

With `pixtrack`, you can:
1. Create an object-level NeRF and a corresponding SFM model capturing its 3D keypoint structure
2. Run 6-DoF object tracking on a video file and visualize the pose trajectory of the object with respect to the camera
---

## Getting Started: One time setup

Follow these steps:

Step1: Update the cuda version in `DockerFile` in line 1 to the version on your machine/server. 

Step2: Setting up the environment. 
```bash
git clone git@github.com:GiantAI/pixtrack.git
cd pixtrack
docker build -t pixtrack .
source run_docker_tolkien.sh
cd pixtrack
source setup.sh
```

---

## Preliminary steps: Create object tracking assets

Before you can run object pose tracking, you need to do the following:

1. Collect object data following the protocol.
2. Create an SfM using the collected images.
3. Train a NeRF using the images and the SfM.
4. Create an object SfM using the NeRF.

Currently, we use environment variables to set paths to the data, sfm, nerf, etc.
An example config file is provided (config/gimble.sh).

Once data is collected (step 1), and assuming the paths are set in the config file, run the following (steps 2-4):
```bash
source config/gimble.sh 
python3 scripts/run_reconstruction.py --images_path $PIXSFM_DATASETS/$OBJECT/ --outputs_path $PIXSFM_OUTPUTS/$OBJECT 
source train_ingp_nerf.sh 
python3 scripts/create_nerf_dataset_and_sfm.py
python3 scripts/augment_sfm.py
```

---

## Run object tracking
To run object tracking, do this:

```bash
cd ~/pixtrack
python3 pixtrack/pose_trackers/pixloc_tracker_r9.py --query <path to directory with query images> --out_dir <path to output directory>
python3 pixtrack/visualization/run_vis_on_poses.py --out_dir <path to output directory containing object tracking results>
```

To create a video from a folder of images, do this:
```
cd <path to output dirctory containing images>
ffmpeg -start_number 1 -pattern_type glob -i '*.jpg' -c:v libx264 -vf "fps=30,format=yuv420p"  overlay.mp4
```
