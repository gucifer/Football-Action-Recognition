import torch
import os
import torchvision.models as models
import cv2
import numpy as np
from tqdm import tqdm
from torchvision.models.feature_extraction import create_feature_extractor
from multiprocessing import Pool
from math import ceil
from torch.utils.data import Dataset, DataLoader


class FrameData(Dataset):

    def __init__(self, game) -> None:
        super().__init__()

        all_game_frames_path = "/Users/srv/Documents/Git/GaTech/dl_cse_7643/final_project/code/data/frames"
        # feature_path = "/Users/srv/Documents/Git/GaTech/dl_cse_7643/final_project/code/data/vit_features"

        # game_features_path = os.path.join(feature_path, game)
        game_frames_path = os.path.join(all_game_frames_path, game)
        frames = os.listdir(game_frames_path)
        frames = [frame for frame in frames if frame[-4:] == ".jpg"]
        frames = sorted(frames, key = lambda x: int(x[:-4]))
        all_frame_attrs = []
        for frame in frames:
            frame_path = os.path.join(game_frames_path, frame)
            frame_attr = cv2.imread(frame_path)
            all_frame_attrs.append(frame_attr)
        all_frame_attrs = torch.tensor(np.asarray(all_frame_attrs)).permute(0, 3, 1, 2)
        transformer_func = model_weights.transforms()
        self.transformed_all_attrs = transformer_func(all_frame_attrs)
    
    def __getitem__(self, index):
        return self.transformed_all_attrs[index]

    def __len__(self):
        return len(self.transformed_all_attrs)



# def model_run(batch_data):
#     model_weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
#     model = models.vit_b_16(weights=model_weights, progress=True).to(device)
#     model.eval()
#     return_nodes = {"encoder.layers.encoder_layer_11": "encode11"}
#     feat_model = create_feature_extractor(model, return_nodes).to(device)
#     features = feat_model(batch_data)
#     return features["encode11"][:, 0]



if __name__ == "__main__":

    if torch.backends.mps.is_available():
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    all_game_frames_path = "/Users/srv/Documents/Git/GaTech/dl_cse_7643/final_project/code/data/frames"
    feature_path = "/Users/srv/Documents/Git/GaTech/dl_cse_7643/final_project/code/data/vit_features"
    batch_size = 50
    workers = 1

    model_weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
    model = models.vit_b_16(weights=model_weights, progress=True).to(device)
    model.eval()
    return_nodes = {"encoder.layers.encoder_layer_11": "encode11"}
    feat_model = create_feature_extractor(model, return_nodes)
    game_list = os.listdir(all_game_frames_path)
    game_list.reverse()
    batch_per_worker = batch_size//workers
    num_batch_iter = ceil(batch_size/batch_per_worker)

    for game in tqdm(game_list):

        extracted_vit_features = os.listdir(feature_path)
        extracted_vit_features = [vit_feat[:-4] for vit_feat in extracted_vit_features]
        if game == ".DS_Store" or game in extracted_vit_features:
            continue

        dataset = FrameData(game)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers)
        # loader = DataLoader(dataset, batch_size=batch_size)

        game_features_path = os.path.join(feature_path, game)

        B, C, H, W = dataset.transformed_all_attrs.shape
        all_features = []
        with torch.no_grad():
            for it, trans_attrs in enumerate(tqdm(loader)):
            # for it in tqdm(range(0, B, batch_size)):
                # pooled_batched_data = []
                # for i in range(num_batch_iter):
                #     cur_ini_batch = it+i*batch_per_worker
                #     cur_end_batch = cur_ini_batch + batch_per_worker
                #     batch_data = dataset.transformed_all_attrs[cur_ini_batch:cur_end_batch, :, :, :].to(device)
                #     pooled_batched_data.append(batch_data)
                    
                # with Pool(workers) as model_runner:
                #     pooled_features = model_runner.map(model_run, pooled_batched_data)
                trans_attrs = trans_attrs.to(device)
                features = feat_model(trans_attrs)
                encode_feats = features["encode11"][:, 0]
                all_features.append(encode_feats.cpu())
                # if it==64*3: break
        all_features = np.concatenate(all_features)
        np.save(os.path.join(game_features_path), all_features)
        