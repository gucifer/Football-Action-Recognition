import torch
import os
import torchvision.models as models
import numpy as np
from tqdm import tqdm
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import DataLoader
from FrameData import *
import argparse
from ModifiedVideoDownloader import *
from FrameExtractorCropper import *



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="ViT Feature Extractor")
    parser.add_argument("-v", "--videos_path", default="videos", help="videos downloader")
    parser.add_argument("-f", "--frames_path", default="frames", help="frames directory")
    parser.add_argument("-s", "--save_path", default="vit_features", help="vit features save directory")
    parser.add_argument("-w", "--workers", default=1, help="number of workers", type=int)
    parser.add_argument("-b", "--batch_size", default=50, help="batch size", type=int)
    parser.add_argument("-r", "--run_reverse_list", default=False, help="extract features in reverse order of gameslist", type=bool)
    args = parser.parse_args()

    print("Downloading Videos...")
    mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=args.videos_path)
    mySoccerNetDownloader.password = input("Password for videos? (Please refer to the report if you don't have this):\n")

    mySoccerNetDownloader.downloadGames(files=["1_224.mkv", "2_224p.mkv", "Labels-v2.json"], split=["train","valid","test","challenge"])  # download Features



    print("Extracting Frames...")
    frame_extractor = FrameExtractorCropper()
    frame_extractor.extract_crop_frames(args.videos_path, args.frames_path)


    print("Extracting Features...")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    all_game_frames_path = args.frames_path
    feature_path = args.save_path
    batch_size = args.batch_size
    workers = args.workers

    model_weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
    model = models.vit_b_16(weights=model_weights, progress=True).to(device)
    model.eval()
    return_nodes = {"encoder.layers.encoder_layer_11": "encode11"}
    feat_model = create_feature_extractor(model, return_nodes)
    game_list = os.listdir(all_game_frames_path)
    assert len(game_list) > 0, "Empty Game List"
    os.makedirs(feature_path, exist_ok=True)
    if args.run_reverse_list:
        game_list.reverse()

    for game in tqdm(game_list):

        extracted_vit_features = os.listdir(feature_path)
        game_features_path = os.path.join(feature_path, game)
        
        extracted_vit_features = [vit_feat[:-4] for vit_feat in extracted_vit_features]
        if game == ".DS_Store" or game in extracted_vit_features:
            continue
        else:
            np.save(os.path.join(game_features_path), np.array([]))

        dataset = FrameData(game, all_game_frames_path)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers)
        
        all_features = []
        with torch.no_grad():
            for it, trans_attrs in enumerate(tqdm(loader)):
                transformer_func = model_weights.transforms()
                transformed_trans_attrs = transformer_func(trans_attrs)
                transformed_trans_attrs = transformed_trans_attrs.to(device)
                features = feat_model(transformed_trans_attrs)
                encode_feats = features["encode11"][:, 0]
                all_features.append(encode_feats.cpu())
                
        all_features = np.concatenate(all_features)
        np.save(os.path.join(game_features_path), all_features)