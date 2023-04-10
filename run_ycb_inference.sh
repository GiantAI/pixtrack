ROOT=/home/wayve/prajwal/pixtrack
python3 pixtrack/pose_trackers/pixloc_tracker_ycb.py --object_path $1 --query $2 --out_dir $3 --use_depth
python3 pixtrack/visualization/run_vis_on_poses.py --object_path $1 --out_dir $3 --pose_error
cd $3/results
ffmpeg -start_number 1 -pattern_type glob -i '*.png' -c:v libx264 -vf "fps=30,format=yuv420p"  ../overlay.mp4
cd $ROOT
