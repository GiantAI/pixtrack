# PixTrack
A 6DOF object pose tracker based on ingp, hloc, pixloc and pixsfm lines of work.

With `pixtrack`, you can:

1. Create an object NeRF and a corresponding SFM model.
2. Run 6DOF object tracking on a video file and visualize the same.

---

## Getting Started: One time setup

Follow these steps:

```bash
ssh 10.0.2.113
git clone git@bitbucket.org:ai_giant_global/pixtrack.git
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
python3 run_reconstruction.py --images_path $PIXSFM_DATASETS/$OBJECT/ --outputs_path $PIXSFM_OUTPUTS/$OBJECT 
source train_ingp_nerf.sh 
python3 create_nerf_dataset_and_sfm.py
python3 augment_sfm.py
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