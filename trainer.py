import os

args = {
    "labels_path": "G:\.shortcut-targets-by-id\\1H509_zeV7bta5BudwCznjP0EWrCa_LCJ\deep_learning_alphas_final_project\data\\assets",
    "batch-size": "24",
    "lr": "2e-3",
    "out_dir": "saved",
    "workers": "1",
    "n_download": "500",
}

cmd = "python -u train_pca.py"
args = " ".join(["--"+k+" "+v for k, v in args.items()])
final_cmd = cmd + " " + args

os.system(final_cmd)