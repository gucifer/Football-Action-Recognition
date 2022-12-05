import os
import getpass

args = {
    "srv_labels_path": "/Users/srv/Documents/Cloud/Google\ Drive\ -\ Default/deep_learning_alphas_final_project/data/assets",
    "vbh_labels_path": "G:\.shortcut-targets-by-id\\1H509_zeV7bta5BudwCznjP0EWrCa_LCJ\deep_learning_alphas_final_project\data\\assets",
    "batch-size": "24",
    "lr": "2e-4",
    "out_dir": "saved",
    "workers": "4",
    "n_download": "500",
    # "resume": "saved/checkpoint.pth.tar"
}
if getpass.getuser() == "srv":
    del args["vbh_labels_path"]
    args["labels_path"] = args.pop("srv_labels_path")
else:
    del args["srv_labels_path"]
    args["labels_path"] = args.pop("vbh_labels_path")


cmd = "python -u train_pca.py"
args = " ".join(["--"+k+" "+v for k, v in args.items()])
final_cmd = cmd + " " + args

os.system(final_cmd)