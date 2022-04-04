class PoseTracker:
    def __init__(self):
        return

    def relocalize(self):
        raise NotImplementedError

    def get_query_frame_iterator(self):
        raise NotImplementedError

    def update_reference_frames(self):
        raise NotImplementedError

    def run_single_frame(self, frame):
        # Get new pose
        pose_success = self.update_pose(frame)

        # Relocalize if needed
        if not pose_success:
            self.relocalize(frame)

        # Update reference frames
        self.update_reference_frames()

    def run(self):
        frame_iterator = self.get_query_frame_iterator()
        for frame in frame_iterator:
            self.run_single_frame(frame)
        return

