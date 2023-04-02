python3 scripts/run_reconstruction.py --images_path $1
source train_ingp_nerf.sh $1
python3 scripts/create_nerf_dataset_and_sfm.py --object_path $1 --object_aabb "$2"
python3 scripts/augment_sfm.py --object_path $1

