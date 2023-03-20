OBJECT="gimble_mar12_2023"
SNAPSHOTS_ROOT="/home/wayve/data/pixtrack/instant-ngp/snapshots"
PIXSFM_DATASETS="/home/wayve/data/pixtrack/pixel-perfect-sfm/datasets"
PIXSFM_OUTPUTS="/home/wayve/data/pixtrack/pixel-perfect-sfm/outputs"
PIXTRACK_OUTPUTS="/home/wayve/data/pixtrack/outputs"
SNAPSHOT_PATH=$SNAPSHOTS_ROOT/$OBJECT

export OBJECT=$OBJECT
export SNAPSHOTS_ROOT=$SNAPSHOTS_ROOT
export PIXSFM_DATASETS=$PIXSFM_DATASETS
export PIXSFM_OUTPUTS=$PIXSFM_OUTPUTS
export SNAPSHOT_PATH=$SNAPSHOT_PATH
export PIXTRACK_OUTPUTS=$PIXTRACK_OUTPUTS
export OBJ_AABB="[[0.302, -0.386, 0.209], [0.735, 0.108, 0.554]]"
export UPRIGHT_REF_IMG="mapping/IMG_9531.png"
export OBJ_CENTER="[0.1179, 1.1538, 1.3870]"

if [ -d $SNAPSHOT_PATH ] 
then
    echo "Snapshot directory exists." 
else
    echo "Snapshot directory does not exist, creating one."
    mkdir -p $SNAPSHOT_PATH
fi

echo "Done setting up paths."

