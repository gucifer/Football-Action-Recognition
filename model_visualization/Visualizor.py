from torch.utils.data import DataLoader

from Visualization.FullModel import FullModel
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)
import os
from math import ceil
import numpy as np

from Visualization.vit_feat_ext import FrameData

if __name__ == "__main__":
    model = FullModel()
    all_game_frames_path = "/Users/srv/Documents/Git/GaTech/dl_cse_7643/final_project/code/data/frames"
    feature_path = "/Users/srv/Documents/Cloud/Georgia Institute of Technology/Alphas - Documents/Project Data/vit_features"
    batch_size = 50
    workers = 1

    game_list = os.listdir(all_game_frames_path)
    # game_list.reverse()
    batch_per_worker = batch_size // workers
    num_batch_iter = ceil(batch_size / batch_per_worker)

    for game in game_list:

        extracted_vit_features = os.listdir(feature_path)
        game_features_path = os.path.join(feature_path, game)

        extracted_vit_features = [vit_feat[:-4] for vit_feat in extracted_vit_features]
        if game == ".DS_Store" or game in extracted_vit_features:
            continue
        else:
            np.save(os.path.join(game_features_path), np.array([]))

        dataset = FrameData(game)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers)

        B, C, H, W = dataset.transformed_all_attrs.shape
        ig = IntegratedGradients(model)
        attributions, delta = ig.attribute(loader, baseline, target=0, return_convergence_delta=True)
        break