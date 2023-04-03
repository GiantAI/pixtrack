# PixTrack

A Computer Vision method for Object Tracking which uses NeRF templates and feature-metric alignment to robustly track the 6DoF pose of a known object.

[![PixTrack](https://img.youtube.com/vi/yQtCUS3i9nc/0.jpg)](https://www.youtube.com/watch?v=yQtCUS3i9nc)

With `pixtrack`, you can:
1. Create an object-level NeRF and a corresponding SFM model capturing its 3D keypoint structure
2. Run 6-DoF object tracking on a video file and visualize the pose trajectory of the object with respect to the camera
---

## Getting Started: One time setup

Follow these steps:

Step1: Update the cuda version in [pixtrack/DockerFile](https://github.com/GiantAI/pixtrack/blob/main/Dockerfile#L1) and [setup.sh](https://github.com/GiantAI/pixtrack/blob/main/setup.sh#L28) to the version on your machine/server. 

Step2: Setting up the environment. 
```bash
git clone git@github.com:GiantAI/pixtrack.git
cd pixtrack
docker build -t pixtrack .
source run_docker.sh
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

Most phone images are in HEIC format. To convert HEIC files to png files, do this:
```
sudo apt-get install libheif-examples
cd <directory_containing_heic_files>
for file in *.HEIC; do heif-convert $file ${file/%.HEIC/.png}; done
```

Once data is collected (step 1), run the following (steps 2-4):
```bash
source images_pipeline.sh <path_to_images> <object_aabb>
```

An example dataset of the `premier_protein` object can be found [here](https://drive.google.com/drive/folders/131AnpOUKmA2hQmHMFZO5JdsFy6JYojME?usp=sharing) 

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
