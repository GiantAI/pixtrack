ROOT=$PWD
#python3 pixtrack/pose_trackers/pixloc_tracker_r9.py --object_path $1 --query $2 --out_dir $3 --debug 1
python3 pixtrack/visualization/run_vis_on_poses.py --object_path $1 --out_dir $3 --obj_aabb "$OBJ_AABB"
cd $3/results
ffmpeg -start_number 1 -pattern_type glob -i '*.jpg' -c:v libx264 -vf "fps=30,format=yuv420p"  ../overlay.mp4
cd $ROOT
