python3 scripts/run_reconstruction.py --images_path $PIXSFM_DATASETS/$OBJECT/ --outputs_path $PIXSFM_OUTPUTS/$OBJECT 
source train_ingp_nerf.sh 
python3 scripts/create_nerf_dataset_and_sfm.py
python3 scripts/augment_sfm.py
