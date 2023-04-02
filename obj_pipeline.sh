python3 scripts/create_sfm_from_obj.py --mesh_path $1/textured.obj
source train_ingp_nerf.sh $1
python3 scripts/augment_sfm.py --object_path $1
