import numpy as np
from tqdm import tqdm
from collections import defaultdict
from hloc.utils.read_write_model import read_model

def extract_covisibility(model_path):
    cameras, images, points3D = read_model(model_path)
    pairs = []
    covis_all = {}
    for image_id, image in tqdm(images.items()):
        matched = image.point3D_ids != -1
        points3D_covis = image.point3D_ids[matched]

        covis = defaultdict(int)
        for point_id in points3D_covis:
            for image_covis_id in points3D[point_id].image_ids:
                if image_covis_id != image_id:
                    covis[image_covis_id] += 1

        if len(covis) == 0:
            continue

        covis_ids = np.array(list(covis.keys()))
        covis_all[image_id] = covis
    return covis_all
