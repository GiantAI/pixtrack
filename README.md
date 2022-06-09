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
docker run -it --rm -p 8090:8090 \
				-e USER="$USER" \
				-e HOME="/home/$USER" \
				-w /home/$USER \
				-v /home/$USER/:/home/$USER/ \
				-v ~/.ssh:/root/.ssh \
				--network host \
				--gpus '"device=0"' \
				--shm-size=256gb \
				pixtrack \
				bash
cd pixtrack
source setup.sh
python3 -m pixloc.download --select checkpoints
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
cd ~/pixtrack
source setup.sh
source config/gimble.sh 
python3 run_reconstruction.py 
source train_ingp_nerf.sh 
python3 create_nerf_dataset_and_sfm.py
```

---

## Run object tracking
To run object tracking, do this:

```bash
cd ~/pixtrack
python3 pixtrack/pose_trackers/pixloc_tracker_r6.py --query <path to directory with query images> --out_dir <path to output directory>
python3 pixtrack/visualization/run_vis_on_poses.py --out_dir <path to output directory containing object tracking results>
```
