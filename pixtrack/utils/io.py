import glob
import os
from pixloc.pixlib.datasets.view import read_image
import numpy as np
import tqdm
import ycbvideo
from pathlib import Path
from pixloc.pixlib.geometry import Pose, Camera as PixCamera
from hloc.utils.read_write_model import Camera as ColCamera
from pixtrack.utils.pytorch3d_render_utils import create_colmap_camera

class YCBVideoIterator:
    def __iter__(self):
        return self

    def __init__(
            self, 
            object_path,
            expression='7/:20', 
            ycb_path='/data/ycb/'):
        self.ycb_root = Path(ycb_path)
        loader = ycbvideo.Loader(ycb_path)
        self.frames = loader.frames([expression])
        class_map = {
                '003_cracker_box': 2, 
                '004_sugar_box': 3,
                '006_mustard_bottle': 5,
                '021_bleach_cleanser': 12,
                '035_power_drill': 15,
                }
        self.object_id = class_map[object_path.name]
        self.idx = 0

    def __len__(self):
        return len(self.frames)

    def __next__(self):
        if self.idx > len(self) - 1:
            raise StopIteration
        query = self.frames[self.idx]
        sequence = query.description.frame_sequence
        frame = query.description.frame
        path = self.ycb_root / 'data' / sequence / f'{frame}-color.png'
        query_image = read_image(path).astype(np.float32)
        semseg = (query.label == self.object_id).astype(np.float32)[:, :, np.newaxis]
        #query_image = query_image * semseg

        intrinsics = query.meta['intrinsic_matrix']
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        cx, cy = 319.5000, 239.5000
        k1 = 0.
        H, W, _ = query.color.shape

        pose_idx = int(np.argwhere(query.meta['cls_indexes'].squeeze() == self.object_id).squeeze())

        pose = query.meta['poses'][:, :, pose_idx]
        R = pose[:, :3]
        T = pose[:, 3]
        pixpose = Pose.from_Rt(R, T)
        camera = ColCamera(
                id=1,
                model="OPENCV",
                width=W,
                height=H,
                params=np.array([fx, fy, cx, cy]),
            )
        pixcamera = PixCamera.from_colmap(camera)

        self.idx += 1
        return path, query_image, pixpose, pixcamera


class ImagePathIterator:
    def __iter__(self):
        return self

    def __init__(self, path, max_frames=10):
        assert os.path.isdir(path)
        jpg_paths = glob.glob(os.path.join(path, '*.jpg'))
        png_paths = glob.glob(os.path.join(path, '*.png'))
        image_paths = jpg_paths + png_paths
        image_paths = sorted(image_paths)
        image_paths = image_paths[:max_frames]
        self.image_paths = image_paths
        self.idx = 0

    def __len__(self):
        return len(self.image_paths)

    def __next__(self):
        if self.idx > len(self) - 1:
            raise StopIteration
        path =  self.image_paths[self.idx]
        self.idx += 1
        return path

class ImageIterator:
    def __iter__(self):
        return self

    def __init__(self, path, max_frames=100):
        assert os.path.isdir(path)
        jpg_paths = glob.glob(os.path.join(path, '*.jpg'))
        png_paths = glob.glob(os.path.join(path, '*.png'))
        image_paths = jpg_paths + png_paths
        image_paths = sorted(image_paths)
        image_paths = image_paths[:max_frames]
        self.image_paths = image_paths
        print('Reading query images')
        self.images = []
        print(len(image_paths))
        for path in tqdm.tqdm(image_paths):
            self.images.append(read_image(path).astype(np.float32))
        self.idx = 0

    def __len__(self):
        return len(self.image_paths)

    def __next__(self):
        if self.idx > len(self) - 1:
            raise StopIteration
        image =  self.images[self.idx]
        path = self.image_paths[self.idx]
        self.idx += 1
        return path, image

