import os
if __name__ == "__main__":
    all_game_frames_path = "/Users/srv/Documents/Cloud/Georgia Institute of Technology/Alphas - Documents/Project Data/frames"
    feature_path = "/Users/srv/Documents/Cloud/Georgia Institute of Technology/Alphas - Documents/Project Data/vit_features_test_igore"
    os.system(f"python -u feature_extractor_main.py -f {all_game_frames_path} -s {feature_path}")