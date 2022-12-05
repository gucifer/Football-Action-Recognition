
import os

import cv2
import numpy as np
from tqdm import tqdm


    



if __name__ == "__main__":


    all_game_frames_path = "/Users/srv/Documents/Git/GaTech/dl_cse_7643/final_project/code/data/frames"
    feature_path = "/Users/srv/Documents/Git/GaTech/dl_cse_7643/final_project/code/data/np_features"
    


    game_list = os.listdir(all_game_frames_path)
    

    for game in tqdm(game_list):
        if game == ".DS_Store": continue
        game_frames_path = os.path.join(all_game_frames_path, game)
        frames = os.listdir(game_frames_path)
        frames = [frame for frame in frames if frame[-4:] == ".jpg"]
        frames = sorted(frames, key = lambda x: int(x[:-4]))
        all_frame_attrs = []
        for frame in frames:
            frame_path = os.path.join(game_frames_path, frame)
            frame_attr = cv2.imread(frame_path)
            all_frame_attrs.append(frame_attr)
        all_frame_attrs = np.asarray(all_frame_attrs)
        game_features_path = os.path.join(feature_path, game)
        np.save(os.path.join(game_features_path), all_frame_attrs)
        