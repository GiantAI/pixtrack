import glob
import os

class ImagePathIterator:
    def __iter__(self):
        return self

    def __init__(self, path):
        assert os.path.isdir(path)
        jpg_paths = glob.glob(os.path.join(path, '*.jpg'))
        png_paths = glob.glob(os.path.join(path, '*.png'))
        image_paths = jpg_paths + png_paths
        image_paths = sorted(image_paths)
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

