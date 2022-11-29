import os

args = {
    "labels_path": "data/assets",
    "batch-size": "24",
    "lr": "2e-3",
    "out_dir": "saved",
    "workers": "2",
    "n_download": "3",
}

cmd = "python -u train_pca.py"
args = " ".join(["--"+k+" "+v for k, v in args.items()])
final_cmd = cmd + " " + args

os.system(final_cmd)