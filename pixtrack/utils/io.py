import glob
import os
from pixloc.pixlib.datasets.view import read_image
import numpy as np
import tqdm

class ImagePathIterator:
    def __iter__(self):
        return self

    def __init__(self, path, max_frames=None):
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

    def __init__(self, path, max_frames=None):
        assert os.path.isdir(path)
        jpg_paths = glob.glob(os.path.join(path, '*.jpg'))
        png_paths = glob.glob(os.path.join(path, '*.png'))
        image_paths = jpg_paths + png_paths
        image_paths = sorted(image_paths)
        image_paths = image_paths[:max_frames]
        self.image_paths = image_paths
        print('Reading query images')
        self.images = []
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

