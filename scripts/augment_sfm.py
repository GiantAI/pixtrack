import argparse
import glob
import os
from pathlib import Path
import shutil
import sys

from hloc.triangulation import import_features, import_matches
from hloc import pairs_from_exhaustive
import pycolmap

from pixtrack.utils.hloc_utils import add_rotation_augmentation_to_features_and_matches, augment_images_and_points3D, create_db_from_model


def main(output):
    sfm_dir = output / 'aug_sfm'
    reference_model = output / 'sfm'
    images = output
    features = output / 'features.h5'
    matches = output / 'matches.h5'
    sfm_pairs = output / 'pairs-sfm.txt'

    sfm_dir.mkdir(parents=True, exist_ok=True)

    image_list = glob.glob(str(images / 'mapping/*.png'))
    image_list = sorted(['/'.join(x.rsplit('/', 2)[-2:]) for x in image_list])

    # Get image dict
    print('Adding rotation augmentation')
    image_dict = add_rotation_augmentation_to_features_and_matches(image_list, images, features, matches, save_images=True)

    # Augment images and 3d points
    print('Creating new sfm binaries')
    augmented_images = augment_images_and_points3D(output, image_dict)

    # Create new sfm database
    database = sfm_dir / 'database.db'
    reference = pycolmap.Reconstruction(reference_model)
    image_ids = create_db_from_model(reference, database, augmented_images)

    # Import features and matches
    ref_imgs = image_list + [image_dict[x][y] for x in image_dict for y in image_dict[x]]
    pairs_from_exhaustive.main(sfm_pairs, image_list=image_list)
    original_ids = {x: image_ids[x] for x in image_ids if image_ids[x] <= len(image_list)}
    import_features(original_ids, database, features)
    import_matches(original_ids, database, sfm_pairs, matches,
                   min_match_score=None, skip_geometric_verification=False)
    print('Done!')

if __name__ == '__main__':
    obj = Path(os.environ['OBJECT'])
    object_path = Path(os.environ['OBJECT_PATH'])
    rout = object_path / 'pixtrack/nerf_sfm'
    aout = object_path / 'pixtrack/aug_nerf_sfm'

    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_output', type=Path, default=rout)
    parser.add_argument('--aug_output', type=Path, default=aout)
    args = parser.parse_args()

    if not os.path.isdir(aout):
        print('Copying reference sfm')
        shutil.copytree(args.ref_output, args.aug_output)
        print('Done: Copying reference sfm')
    main(args.aug_output)
    with open(str(aout / '*_with_intrinsics.txt'), 'w'):
        pass
