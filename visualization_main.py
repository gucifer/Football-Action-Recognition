import os
import torch
from torch.utils.data import DataLoader
from model_visualization.FullModel import FullModel
import captum.attr as ct_attr
import os
from training_modules.dataset import SoccerNetClips, INVERSE_EVENT_DICTIONARY_V2
from tqdm import tqdm
from training_modules.model import Model
from training_modules.attention_model import AttentionModel
import argparse
from model_visualization.captum_utils import visualize_attr_maps





if __name__ == "__main__":


    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    parser = argparse.ArgumentParser(description="Model Visualization")
    parser.add_argument("--frames_path", default="frames", help="frames directory")
    parser.add_argument("--SoccerNet_path", default="assets", help="assets directory")
    parser.add_argument("--artefacts_path", default="visualization", help="visualization save directory")
    parser.add_argument("--batch_size", default=30, help="batch size", type=int)
    parser.add_argument("--num_games", default=4, help="number of games to read", type=int)

    args = parser.parse_args()
    visualizers = [
        # ct_attr.GradientShap,
        # ct_attr.DeepLift,
        # ct_attr.DeepLiftShap,
        # ct_attr.IntegratedGradients,
        # ct_attr.LayerConductance,
        # ct_attr.NeuronConductance,
        # ct_attr.NoiseTunnel,
        ct_attr.LayerGradCam,
        # ct_attr.LayerConductance,
    ]
    
    saved_models = os.listdir(args.artefacts_path)
    dataset = SoccerNetClips(path=args.SoccerNet_path, features="frames", custom_feature_path=args.frames_path, split="test", n=args.num_games)
    data_loader = DataLoader(dataset, batch_size=args.batch_size)
    # for model in saved_models:
    model = "NetVLAD++_vit_CE"
    # if model == ".DS_Store": continue
    model_name, feat, loss, *_ = model.split("_")
    print(f"Running: {model_name}.{feat}.{loss}")
    # if feat != "vit": continue
    runs = os.listdir(os.path.join(args.artefacts_path, model))
    for run in runs:
        if "mps" in run:
            if model_name == "NetVLAD++":
                classification_model = Model(input_size=768, num_classes=17, window_size=15, vocab_size = 64, framerate=2, pool="NetVLAD++")
            elif model_name == "ATTENTION":
                classification_model = AttentionModel(feature_size=768, num_frames=30, num_heads=8, num_classes = 17, dropout=0.45)
            else:
                continue
            classification_model_checkpoint = torch.load(os.path.join(args.artefacts_path, model, run, "model.pth.tar"))
            classification_model.load_state_dict(classification_model_checkpoint['state_dict'])
            full_model = FullModel(classification_model, device)
            full_model.eval()
            for param in full_model.parameters():
                param.requires_grad = True
            for visualizer in visualizers:
                visual_save_path = os.path.join(args.artefacts_path, model, run, "_".join(visualizer.get_name().split(" ")))
                os.makedirs(visual_save_path, exist_ok=True)
                # algo = visualizer(full_model, full_model.classification_model.fc)
                # algo = visualizer(full_model, full_model.feature_model.encoder.layers.encoder_layer_1)
                algo = visualizer(full_model, full_model.feature_model.conv_proj)
                for it, (attrs, labels) in enumerate(tqdm(data_loader)):
                    transformed_attrs = full_model.preprocess(attrs).to(device)
                    out = full_model(transformed_attrs)
                    max_out = torch.argmax(out, dim=1)
                    pred_label = max_out.cpu().detach().numpy()[0] - 1
                    labels = labels.to(device)
                    cp_labels = torch.argmax(labels, dim=1)[::30]
                    attribution = algo.attribute(transformed_attrs, target = cp_labels)
                    labels = torch.argmax(labels, dim=1) - 1
                    actual_label = labels.cpu().detach().numpy()[0]
                    attrs = attrs.permute(0, 2, 3, 1)
                    correctness = "correct" if actual_label == pred_label else "incorrect"
                    if INVERSE_EVENT_DICTIONARY_V2[actual_label] == "Background":
                        continue
                    else:
                        halt=1
                    classif = "-".join(INVERSE_EVENT_DICTIONARY_V2[actual_label].split(" ")) + "_" + correctness + "_" + "-".join(INVERSE_EVENT_DICTIONARY_V2[pred_label].split(" "))
                    visualize_attr_maps(os.path.join(visual_save_path, f"{it}_{classif}.png"), attrs, INVERSE_EVENT_DICTIONARY_V2[actual_label], [attribution.cpu()], [visualizer.get_name()], N = 5)
