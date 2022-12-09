import os
import logging
from datetime import datetime
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch

from dataset import SoccerNetClips, SoccerNetClipsTesting #,SoccerNetClipsOld
from model import Model
from attention_model import AttentionModel
from RMSNET import RMSNetModel
from train import trainer, test, testSpotting
from loss import NLLLoss, reweight, FocalLoss


def main(args):


    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logging.info("Parameters:")
    for arg in vars(args):
        logging.info(arg.rjust(15) + " : " + str(getattr(args, arg)))

    # create dataset
    # args.test_only = True
    if not args.test_only:
        dataset_Train = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split=args.split_train, version=args.version, framerate=args.framerate, window_size=args.window_size, custom_feature_path=args.custom_feature_path)
        dataset_Valid = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split=args.split_valid, version=args.version, framerate=args.framerate, window_size=args.window_size, custom_feature_path=args.custom_feature_path)
        dataset_Valid_metric  = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split=args.split_valid, version=args.version, framerate=args.framerate, window_size=args.window_size, custom_feature_path=args.custom_feature_path)
    dataset_Test  = SoccerNetClipsTesting(path=args.SoccerNet_path, features=args.features, split=args.split_test, version=args.version, framerate=args.framerate, window_size=args.window_size, custom_feature_path=args.custom_feature_path)

    if args.feature_dim is None:
        args.feature_dim = dataset_Test[0][1].shape[-1]
        print("feature_dim found:", args.feature_dim)

    # create model
    ml_model_name = args.model_name.split("_")[0]

    if ml_model_name == "ATTENTION":
        model = AttentionModel(feature_size=args.feature_dim,
                        num_frames=args.window_size*args.framerate, 
                        num_heads=8,
                        num_classes = dataset_Test.num_classes,
                        dropout=0.45).to(device)

    elif ml_model_name == "RMSNet":
        model = RMSNetModel(feature_size=args.feature_dim,
                        num_frames=args.window_size*args.framerate, 
                        num_classes = dataset_Test.num_classes,
                        dropout=0.45).to(device)
    else:
        model = Model(weights=args.load_weights, input_size=args.feature_dim,
                    num_classes=dataset_Test.num_classes, window_size=args.window_size, 
                    vocab_size = args.vocab_size,
                    framerate=args.framerate, pool=args.pool).to(device)

    logging.info(model)
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    parameters_per_layer  = [p.numel() for p in model.parameters() if p.requires_grad]
    logging.info("Total number of parameters: " + str(total_params))

    # create dataloader
    if not args.test_only:
        train_loader = torch.utils.data.DataLoader(dataset_Train,
            batch_size=args.batch_size, shuffle=True,)
            # num_workers=args.max_num_worker, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(dataset_Valid,
            batch_size=args.batch_size, shuffle=False,)
            # num_workers=args.max_num_worker, pin_memory=True)

        val_metric_loader = torch.utils.data.DataLoader(dataset_Valid_metric,
            batch_size=args.batch_size, shuffle=False,)
            # num_workers=args.max_num_worker, pin_memory=True)


    # training parameters
    if not args.test_only:
        from collections import Counter
        if "CB" in args.loss_function:
            print("beta = 0.9")
            beta = 0.9
        else:
            beta = 0
        
        if "FL" in args.loss_function:
            print("gamma = 1")
            gamma = 1
        else:
            gamma = 0
        per_cls_weights = reweight(dict(Counter(train_loader.dataset.game_labels.argmax(axis=1))), beta=beta).to(device)
        # print(per_cls_weights)
        # criterion = NLLLoss()
        # criterion = torch.nn.CrossEntropyLoss()
        criterion = FocalLoss(per_cls_weights, gamma=gamma)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, 
                                    betas=(0.9, 0.999), eps=1e-08, 
                                    weight_decay=0, amsgrad=False)


        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=args.patience)

        # start training
        trainer(train_loader, val_loader, val_metric_loader, 
                model, optimizer, scheduler, criterion,
                model_name=args.model_name,
                max_epochs=args.max_epochs, evaluation_frequency=args.evaluation_frequency, device = device)

    # Free up some RAM memory
    if not args.test_only:
        del dataset_Train, dataset_Valid, dataset_Valid_metric, dataset_Test
        del train_loader, val_loader, val_metric_loader

    # For the best model only
    checkpoint = torch.load(os.path.join("models", args.model_name, "model.pth.tar"))
    model.load_state_dict(checkpoint['state_dict'])

    # test on multiple splits [test/challenge]
    for split in args.split_test:
        dataset_Test  = SoccerNetClipsTesting(path=args.SoccerNet_path, features=args.features, split=[split], version=args.version, framerate=args.framerate, window_size=args.window_size, custom_feature_path=args.custom_feature_path)

        test_loader = torch.utils.data.DataLoader(dataset_Test,
            batch_size=1, shuffle=False,)
            # num_workers=1, pin_memory=True)

        results = testSpotting(test_loader, model=model, model_name=args.model_name, NMS_window=args.NMS_window, NMS_threshold=args.NMS_threshold, device=device)
        if results is None:
            continue

        a_mAP = results["a_mAP"]
        a_mAP_per_class = results["a_mAP_per_class"]
        a_mAP_visible = results["a_mAP_visible"]
        a_mAP_per_class_visible = results["a_mAP_per_class_visible"]
        a_mAP_unshown = results["a_mAP_unshown"]
        a_mAP_per_class_unshown = results["a_mAP_per_class_unshown"]

        logging.info("Best Performance at end of training ")
        logging.info("a_mAP visibility all: " +  str(a_mAP))
        logging.info("a_mAP visibility all per class: " +  str( a_mAP_per_class))
        logging.info("a_mAP visibility visible: " +  str( a_mAP_visible))
        logging.info("a_mAP visibility visible per class: " +  str( a_mAP_per_class_visible))
        logging.info("a_mAP visibility unshown: " +  str( a_mAP_unshown))
        logging.info("a_mAP visibility unshown per class: " +  str( a_mAP_per_class_unshown))

    return 

if __name__ == '__main__':


    parser = ArgumentParser(description='context aware loss function', formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--SoccerNet_path',   required=False, type=str,   default="G:\.shortcut-targets-by-id\\1H509_zeV7bta5BudwCznjP0EWrCa_LCJ\\deep_learning_alphas_final_project\\assets_pca",     help='Path for SoccerNet' )
    # parser.add_argument('--features',   required=False, type=str,   default="ResNET_TF2_PCA512.npy",     help='Video features' )
    parser.add_argument('--features',   required=False, type=str,   default="custom_vit",     help='Video features' )

    parser.add_argument('--max_epochs',   required=False, type=int,   default=1000,     help='Maximum number of epochs' )
    parser.add_argument('--load_weights',   required=False, type=str,   default=None,     help='weights to load' )
    parser.add_argument('--model_name',   required=False, type=str,   default="NetVLAD++_vit_run_1",     help='named of the model to save' )
    parser.add_argument('--test_only',   required=False, action='store_true',  help='Perform testing only' )

    parser.add_argument('--split_train', nargs='+', default=["train"], help='list of split for training')
    parser.add_argument('--split_valid', nargs='+', default=["valid"], help='list of split for validation')
    parser.add_argument('--split_test', nargs='+', default=["test", "challenge"], help='list of split for testing')

    parser.add_argument('--version', required=False, type=int,   default=2,     help='Version of the dataset' )
    parser.add_argument('--feature_dim', required=False, type=int,   default=None,     help='Number of input features' )
    parser.add_argument('--evaluation_frequency', required=False, type=int,   default=10,     help='Number of chunks per epoch' )
    parser.add_argument('--framerate', required=False, type=int,   default=2,     help='Framerate of the input features' )
    parser.add_argument('--window_size', required=False, type=int,   default=15,     help='Size of the chunk (in seconds)' )
    parser.add_argument('--pool',       required=False, type=str,   default="NetVLAD++", help='How to pool' )
    parser.add_argument('--vocab_size',       required=False, type=int,   default=64, help='Size of the vocabulary for NetVLAD' )
    parser.add_argument('--NMS_window',       required=False, type=int,   default=30, help='NMS window in second' )
    parser.add_argument('--NMS_threshold',       required=False, type=float,   default=0.0, help='NMS threshold for positive results' )

    parser.add_argument('--batch_size', required=False, type=int,   default=256,     help='Batch size' )
    parser.add_argument('--LR',       required=False, type=float,   default=1e-03, help='Learning Rate' )
    parser.add_argument('--LRe',       required=False, type=float,   default=1e-06
, help='Learning Rate end' )
    parser.add_argument('--patience', required=False, type=int,   default=10,     help='Patience before reducing LR (ReduceLROnPlateau)' )

    parser.add_argument('--GPU',        required=False, type=int,   default="0",     help='ID of the GPU to use' )
    parser.add_argument('--max_num_worker',   required=False, type=int,   default=1, help='number of worker to load data')
    parser.add_argument('--seed',   required=False, type=int,   default=3, help='seed for reproducibility')

    parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')
    parser.add_argument('--custom_feature_path',   required=False, type=str,   default="C:\\Users\\91995\\OneDrive - Georgia Institute of Technology\\vit_features", help='custom feature path')
    parser.add_argument('--loss_function',   required=False, type=str,   default="CE", help='Loss function')

    args = parser.parse_args()

    # for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)

    os.makedirs(os.path.join("models", args.model_name), exist_ok=True)
    log_path = os.path.join("models", args.model_name,
                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log'))
    logging.basicConfig(
        level=numeric_level,
        format=
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ])

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)


    start=time.time()
    logging.info('Starting main function')
    main(args)
    logging.info(f'Total Execution Time is {time.time()-start} seconds')
