OBJECT="bottle_07192022"
SNAPSHOTS_ROOT="/data/pixtrack/instant-ngp/snapshots"
PIXSFM_DATASETS="/data/pixtrack/pixel-perfect-sfm/datasets"
PIXSFM_OUTPUTS="/data/pixtrack/pixel-perfect-sfm/outputs"
PIXTRACK_OUTPUTS="/data/pixtrack/outputs"
SNAPSHOT_PATH=$SNAPSHOTS_ROOT/$OBJECT

export OBJECT=$OBJECT
export SNAPSHOTS_ROOT=$SNAPSHOTS_ROOT
export PIXSFM_DATASETS=$PIXSFM_DATASETS
export PIXSFM_OUTPUTS=$PIXSFM_OUTPUTS
export SNAPSHOT_PATH=$SNAPSHOT_PATH
export PIXTRACK_OUTPUTS=$PIXTRACK_OUTPUTS
export OBJ_AABB="[[0.352, -0.080, 0.391], [0.530, 0.409, 0.566]]"
export UPRIGHT_REF_IMG="mapping/IMG_3297.png"
export OBJ_CENTER="[-0.2311, 0.2281, 0.349]"

if [ -d $SNAPSHOT_PATH ] 
then
    echo "Snapshot directory exists." 
else
    echo "Snapshot directory does not exist, creating one."
    mkdir -p $SNAPSHOT_PATH
fi

echo "Done setting up paths."

