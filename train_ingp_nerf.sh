python3 pixtrack/utils/colmap2ingp.py --images $PIXSFM_DATASETS/$OBJECT/mapping --bin $PIXSFM_OUTPUTS/$OBJECT/ref --out $PIXSFM_DATASETS/$OBJECT/ --aabb_scale 4

python3 instant-ngp/scripts/run.py --scene $PIXSFM_DATASETS/$OBJECT --mode nerf --save_snapshot $SNAPSHOT_PATH/weights.msgpack --n_steps 100000

