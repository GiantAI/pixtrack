OBJECT="sync1"
SNAPSHOTS_ROOT="/data/pixtrack/instant-ngp/snapshots"
PIXSFM_DATASETS="/data/pixtrack/pixel-perfect-sfm/datasets"
PIXSFM_OUTPUTS="/data/pixtrack/pixel-perfect-sfm/outputs"
PIXTRACK_OUTPUTS="/data/pixtrack/outputs"
SNAPSHOT_PATH=$SNAPSHOTS_ROOT/$OBJECT
#"crop": [[0.080, 0.070, 0.189], [0.612, 0.406, 0.703]]

export OBJECT=$OBJECT
export SNAPSHOTS_ROOT=$SNAPSHOTS_ROOT
export PIXSFM_DATASETS=$PIXSFM_DATASETS
export PIXSFM_OUTPUTS=$PIXSFM_OUTPUTS
export SNAPSHOT_PATH=$SNAPSHOT_PATH
export PIXTRACK_OUTPUTS=$PIXTRACK_OUTPUTS

if [ -d $SNAPSHOT_PATH ] 
then
    echo "Snapshot directory exists." 
else
    echo "Snapshot directory does not exist, creating one."
    mkdir -p $SNAPSHOT_PATH
fi

echo "Done setting up paths."

