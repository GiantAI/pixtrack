python3 scripts/create_sfm_from_obj.py --mesh_path $OBJECT_PATH/textured.obj
source train_ingp_nerf.sh 
python3 scripts/create_nerf_dataset_and_sfm.py
python3 scripts/augment_sfm.py
