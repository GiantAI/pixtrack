OBJECT="premier_protein_07132022"
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
export OBJ_AABB="[[0.359, -0.248, 0.047], [0.627, 0.223, 0.574]]"
export UPRIGHT_REF_IMG="mapping/IMG_2520.png"

if [ -d $SNAPSHOT_PATH ] 
then
    echo "Snapshot directory exists." 
else
    echo "Snapshot directory does not exist, creating one."
    mkdir -p $SNAPSHOT_PATH
fi

echo "Done setting up paths."

