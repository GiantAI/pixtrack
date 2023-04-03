import tqdm
import numpy as np


class PoseTracker:
    def __init__(self):
        return

    def relocalize(self):
        raise NotImplementedError

    def refine(self):
        raise NotImplementedError

    def get_query_frame_iterator(self):
        raise NotImplementedError

    def update_reference_ids(self):
        raise NotImplementedError

    def run_single_frame(self, frame):
        # Get new pose
        pose_success = self.refine(frame)

        # Relocalize if needed
        if not pose_success:
            self.relocalize(frame)

        # Update reference images
        self.update_reference_ids()

    def run(self, query_path, max_frames=np.inf):
        frame_iterator = self.get_query_frame_iterator(query_path, max_frames)
        self.pbar = tqdm.tqdm(frame_iterator)
        for frame in self.pbar:
            self.run_single_frame(frame)
        return
