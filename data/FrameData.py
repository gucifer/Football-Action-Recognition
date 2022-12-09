import torch
import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class FrameData(Dataset):

    def __init__(self, game, all_game_frames_path) -> None:
        super().__init__()

        game_frames_path = os.path.join(all_game_frames_path, game)
        frames = os.listdir(game_frames_path)
        frames = [frame for frame in frames if frame[-4:] == ".jpg"]
        frames = sorted(frames, key = lambda x: int(x[:-4]))
        self.all_frame_attrs = []
        for frame in frames:
            frame_path = os.path.join(game_frames_path, frame)
            frame_attr = cv2.imread(frame_path)
            self.all_frame_attrs.append(frame_attr)
        self.all_frame_attrs = torch.tensor(np.asarray(self.all_frame_attrs)).permute(0, 3, 1, 2)
    
    def __getitem__(self, index):
        return self.all_frame_attrs[index]

    def __len__(self):
        return len(self.all_frame_attrs)
        