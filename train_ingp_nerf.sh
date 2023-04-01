python3 pixtrack/utils/colmap2ingp.py --images $1/pixtrack/pixsfm/dataset/mapping --bin $1/pixtrack/pixsfm/outputs/ref --out $1/pixtrack/pixsfm/dataset/ --aabb_scale 4
SNAPSHOT_PATH=$1/pixtrack/instant-ngp/snapshots
if [ -d $SNAPSHOT_PATH ] 
then
    echo "Snapshot directory exists." 
else
    echo "Snapshot directory does not exist, creating one."
    mkdir -p $SNAPSHOT_PATH
fi

echo "Done setting up paths."

python3 instant-ngp/scripts/run.py --scene $1/pixtrack/pixsfm/dataset --mode nerf --save_snapshot $SNAPSHOT_PATH/weights.msgpack --n_steps 10000

