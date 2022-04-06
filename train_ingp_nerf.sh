SNAPSHOTS_PATH="$PROJECT_ROOT/instant-ngp/snapshots/gimble_04MAR2022/"
if [ -d $SNAPSHOTS_PATH ] 
then
    echo "Snapshots directory exists." 
else
    echo "Snapshots directory does not exist."
    mkdir -p $SNAPSHOTS_PATH
fi

python3 instant-ngp/scripts/run.py --scene $PROJECT_ROOT/instant-ngp/data/nerf/gimble_04MAR2022/ --mode nerf --save_snapshot $SNAPSHOTS_PATH/weights.msgpack --n_steps 10000
