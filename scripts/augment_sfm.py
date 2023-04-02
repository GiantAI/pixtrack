import argparse
import glob
import os
from pathlib import Path
import shutil
import sys

from hloc.triangulation import import_features, import_matches
from hloc import pairs_from_exhaustive
import pycolmap

from pixtrack.utils.hloc_utils import (
    add_rotation_augmentation_to_features_and_matches,
    augment_images_and_points3D,
    create_db_from_model,
)


def main(output):
    sfm_dir = output / "aug_sfm"
    reference_model = output / "ref"
    images = output
    features = output / "features.h5"
    matches = output / "matches.h5"
    sfm_pairs = output / "pairs-sfm.txt"

    sfm_dir.mkdir(parents=True, exist_ok=True)

    image_list = glob.glob(str(images / "mapping/*.png"))
    image_list = sorted(["/".join(x.rsplit("/", 2)[-2:]) for x in image_list])

    # Get image dict
    print("Adding rotation augmentation")
    image_dict = add_rotation_augmentation_to_features_and_matches(
        image_list, images, features, matches, save_images=False
    )

    # Augment images and 3d points
    print("Creating new sfm binaries")
    augmented_images = augment_images_and_points3D(output, image_dict)

    # Create new sfm database
    database = sfm_dir / "database.db"
    reference = pycolmap.Reconstruction(reference_model)
    image_ids = create_db_from_model(reference, database, augmented_images)

    # Import features and matches
    ref_imgs = image_list + [
        image_dict[x][y] for x in image_dict for y in image_dict[x]
    ]
    pairs_from_exhaustive.main(sfm_pairs, image_list=image_list)
    original_ids = {
        x: image_ids[x] for x in image_ids if image_ids[x] <= len(image_list)
    }
    import_features(original_ids, database, features)
    import_matches(
        original_ids,
        database,
        sfm_pairs,
        matches,
        min_match_score=None,
        skip_geometric_verification=False,
    )
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_path", type=Path)
    parser.add_argument("--use_nerf_sfm", action="store_true", default=False)
    args = parser.parse_args()
    object_path = args.object_path
    if args.use_nerf_sfm:
        ref_output = object_path / "pixtrack/nerf_sfm"
    else:
        ref_output = object_path / "pixtrack/pixsfm/outputs"
    img_output = object_path / "pixtrack/pixsfm/dataset/mapping"
    aug_output = object_path / "pixtrack/aug_nerf_sfm"

    if not os.path.isdir(aug_output):
        print("Copying reference sfm")
        shutil.copytree(ref_output, aug_output)
        shutil.copytree(img_output, aug_output / "mapping")
        print("Done: Copying reference sfm")
    main(aug_output)
    with open(str(aug_output / "*_with_intrinsics.txt"), "w"):
        pass
