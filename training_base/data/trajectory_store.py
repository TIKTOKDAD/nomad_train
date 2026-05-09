import os
import pickle
import sys

import numpy.core as np_core


sys.modules.setdefault("numpy._core", np_core)
if hasattr(np_core, "multiarray"):
    sys.modules.setdefault("numpy._core.multiarray", np_core.multiarray)


class PickleTrajectoryStore:
    """Cached reader for legacy traj_data.pkl files."""

    def __init__(self, data_folder: str) -> None:
        self.data_folder = data_folder
        self.cache = {}

    def get(self, trajectory_name: str):
        if trajectory_name not in self.cache:
            path = os.path.join(self.data_folder, trajectory_name, "traj_data.pkl")
            with open(path, "rb") as f:
                self.cache[trajectory_name] = pickle.load(f)
        return self.cache[trajectory_name]
