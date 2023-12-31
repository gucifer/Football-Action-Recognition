from torch.utils.data import Dataset
from copy import deepcopy
import numpy as np

import os

import cv2
from tqdm import tqdm

import torch

import logging
import json

from SoccerNet.Downloader import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.Evaluation.utils import AverageMeter, EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V1, INVERSE_EVENT_DICTIONARY_V1



def feats2clip(feats, stride, clip_length, padding = "replicate_last", off=0):
    if padding =="zeropad":
        print("beforepadding", feats.shape)
        pad = feats.shape[0] - int(feats.shape[0]/stride)*stride
        print("pad need to be", clip_length-pad)
        m = torch.nn.ZeroPad2d((0, 0, clip_length-pad, 0))
        feats = m(feats)
        print("afterpadding", feats.shape)
        # nn.ZeroPad2d(2)

    idx = torch.arange(start=0, end=feats.shape[0]-1, step=stride)
    idxs = []
    for i in torch.arange(-off, clip_length-off):
        idxs.append(idx+i)
    idx = torch.stack(idxs, dim=1)

    if padding=="replicate_last":
        idx = idx.clamp(0, feats.shape[0]-1)
    # print(idx)
    return feats[idx,...]


class SoccerNetClips(Dataset):

    def __init__(self, path, features="ResNET_PCA512.npy", split=["train"], version=2, 
                framerate=2, window_size=15, custom_feature_path="C:\\Users\\91995\\OneDrive - Georgia Institute of Technology\\vit_features", n = 500):
        self.path = path
        self.listGames = getListGames(split)
        self.features = features
        self.window_size_frame = window_size*framerate
        self.version = version
        self.custom_feature_path=custom_feature_path
        if version == 1:
            self.num_classes = 3
            self.labels="Labels.json"
        elif version == 2:
            self.dict_event = EVENT_DICTIONARY_V2
            self.num_classes = 17
            self.labels="Labels-v2.json"

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)
        if self.features != "custom_vit":

            downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=split, verbose=False, randomized=True)
        # vit features link # https://gtvault.sharepoint.com/:f:/s/Alphas/Eo8u3Gc5jslBhBV13wYlTL0BrhwqMPWxgC2CBGM2zqO2cg?e=ldhIO5
        logging.info("Pre-compute clips")

        self.game_feats = list()
        self.game_labels = list()

        self.game_frames = list()

        # game_counter = 0
        for it, game in enumerate(tqdm(self.listGames)):
            if self.features == "custom_vit":
                game_feature_path = game.replace(os.path.sep, '_')

                feat_half1 = np.load(os.path.join(self.custom_feature_path, f"{game_feature_path}_1.npy"))
                feat_half1 = feat_half1.reshape(-1, feat_half1.shape[-1])
                feat_half2 = np.load(os.path.join(self.custom_feature_path, f"{game_feature_path}_2.npy"))
                feat_half2 = feat_half2.reshape(-1, feat_half2.shape[-1])


            elif self.features == "frames":
                if it>n:break
                game_feature_path = game.replace(os.path.sep, '_')
                feat_half1_path = os.path.join(self.custom_feature_path, game_feature_path + "_1")
                feat_half2_path = os.path.join(self.custom_feature_path, game_feature_path + "_2")
                frames_1 = os.listdir(feat_half1_path)
                frames_1 = [frame for frame in frames_1 if frame[-4:] == ".jpg"]
                frames_1 = sorted(frames_1, key = lambda x: int(x[:-4]))
                feat_half1 = []
                for frame in frames_1:
                    frame_path = os.path.join(feat_half1_path, frame)
                    frame_attr = cv2.imread(frame_path)
                    feat_half1.append(frame_attr)
                feat_half1 = torch.tensor(np.asarray(feat_half1)).permute(0, 3, 1, 2).detach().numpy()

                frames_2 = os.listdir(feat_half2_path)
                frames_2 = [frame for frame in frames_2 if frame[-4:] == ".jpg"]
                frames_2 = sorted(frames_2, key = lambda x: int(x[:-4]))
                feat_half2 = []
                for frame in frames_2:
                    frame_path = os.path.join(feat_half2_path, frame)
                    frame_attr = cv2.imread(frame_path)
                    feat_half2.append(frame_attr)
                feat_half2 = torch.tensor(np.asarray(feat_half2)).permute(0, 3, 1, 2).detach().numpy()
                frames_feat_1 = deepcopy(feat_half1)
                frames_feat_2 = deepcopy(feat_half2)
                
            else:
                feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features))
                feat_half1 = feat_half1.reshape(-1, feat_half1.shape[-1])
                feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features))
                feat_half2 = feat_half2.reshape(-1, feat_half2.shape[-1])

            feat_half1 = feats2clip(torch.from_numpy(feat_half1), stride=self.window_size_frame, clip_length=self.window_size_frame)
            feat_half2 = feats2clip(torch.from_numpy(feat_half2), stride=self.window_size_frame, clip_length=self.window_size_frame)

            # Load labels
            labels = json.load(open(os.path.join(self.path, game, self.labels)))


            
            
            label_half1 = np.zeros((feat_half1.shape[0], self.num_classes+1), dtype=np.float32)
            label_half1[:,0]=1 # those are BG classes
            label_half2 = np.zeros((feat_half2.shape[0], self.num_classes+1), dtype=np.float32)

            label_half2[:,0]=1 # those are BG classes


            for annotation in labels["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                frame = framerate * ( seconds + 60 * minutes ) 

                if version == 1:
                    if "card" in event: label = 0
                    elif "subs" in event: label = 1
                    elif "soccer" in event: label = 2
                    else: continue
                elif version == 2:
                    if event not in self.dict_event:
                        continue
                    label = self.dict_event[event]

                # if label outside temporal of view
                if half == 1 and frame//self.window_size_frame>=label_half1.shape[0]:
                    continue
                if half == 2 and frame//self.window_size_frame>=label_half2.shape[0]:
                    continue

                if half == 1:
                    label_half1[frame//self.window_size_frame][0] = 0 # not BG anymore
                    label_half1[frame//self.window_size_frame][label+1] = 1 # that's my class

                if half == 2:
                    label_half2[frame//self.window_size_frame][0] = 0 # not BG anymore
                    label_half2[frame//self.window_size_frame][label+1] = 1 # that's my class

            self.game_feats.append(feat_half1)
            self.game_feats.append(feat_half2)
            self.game_labels.append(label_half1)
            self.game_labels.append(label_half2)
            feat_half1 = None
            feat_half2 = None
            label_half1 = None
            label_half2 = None

            if self.features == "frames":
                self.game_frames.append(frames_feat_1)
                self.game_frames.append(frames_feat_2)
                frames_feat_1 = None
                frames_feat_2 = None

        self.game_feats = np.concatenate(self.game_feats)
        self.game_labels = np.concatenate(self.game_labels)
        if self.features == "frames":
            self.game_frames = np.concatenate(self.game_frames)
            self.game_feats = None





    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            clip_feat (np.array): clip of features.
            clip_labels (np.array): clip of labels for the segmentation.
            clip_targets (np.array): clip of targets for the spotting.
        """

        if self.features == "frames":
            return self.game_frames[index, :, :, :], self.game_labels[index//(self.window_size_frame), :]
        else:
            return self.game_feats[index,:,:], self.game_labels[index,:]

    def __len__(self):
        if self.features == "frames":
            return len(self.game_frames)
        else:
            return len(self.game_feats)



class SoccerNetClipsTesting(Dataset):
    def __init__(self, path, features="ResNET_PCA512.npy", split=["test"], version=1, 
                framerate=2, window_size=15, custom_feature_path="C:\\Users\\91995\\OneDrive - Georgia Institute of Technology\\vit_features"):
        self.path = path
        self.listGames = getListGames(split)
        self.features = features
        self.window_size_frame = window_size*framerate
        self.framerate = framerate
        self.version = version
        self.split=split
        self.custom_feature_path=custom_feature_path
        if version == 1:
            self.dict_event = EVENT_DICTIONARY_V1
            self.num_classes = 3
            self.labels="Labels.json"
        elif version == 2:
            self.dict_event = EVENT_DICTIONARY_V2
            self.num_classes = 17
            self.labels="Labels-v2.json"

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)
        if self.features != "custom_vit":
            for s in split:
                if s == "challenge":
                    downloader.downloadGames(files=[f"1_{self.features}", f"2_{self.features}"], split=[s], verbose=False,randomized=True)
                else:
                    downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[s], verbose=False,randomized=True)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            feat_half1 (np.array): features for the 1st half.
            feat_half2 (np.array): features for the 2nd half.
            label_half1 (np.array): labels (one-hot) for the 1st half.
            label_half2 (np.array): labels (one-hot) for the 2nd half.
        """
        # Load features
        if self.features == "custom_vit":

            game_feature_path = self.listGames[index].replace(os.path.sep, '_')

            feat_half1 = np.load(os.path.join(self.custom_feature_path, f"{game_feature_path}_1.npy"))
            feat_half1 = feat_half1.reshape(-1, feat_half1.shape[-1])
            feat_half2 = np.load(os.path.join(self.custom_feature_path, f"{game_feature_path}_2.npy"))
            feat_half2 = feat_half2.reshape(-1, feat_half2.shape[-1])
        else:
            feat_half1 = np.load(os.path.join(self.path, self.listGames[index], "1_" + self.features))
            feat_half1 = feat_half1.reshape(-1, feat_half1.shape[-1])
            feat_half2 = np.load(os.path.join(self.path, self.listGames[index], "2_" + self.features))
            feat_half2 = feat_half2.reshape(-1, feat_half2.shape[-1])


        # Load labels
        label_half1 = np.zeros((feat_half1.shape[0], self.num_classes))
        label_half2 = np.zeros((feat_half2.shape[0], self.num_classes))

        # check if annoation exists
        if os.path.exists(os.path.join(self.path, self.listGames[index], self.labels)):
            labels = json.load(open(os.path.join(self.path, self.listGames[index], self.labels)))

            for annotation in labels["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                frame = self.framerate * ( seconds + 60 * minutes ) 

                if self.version == 1:
                    if "card" in event: label = 0
                    elif "subs" in event: label = 1
                    elif "soccer" in event: label = 2
                    else: continue
                elif self.version == 2:
                    if event not in self.dict_event:
                        continue
                    label = self.dict_event[event]

                value = 1
                if "visibility" in annotation.keys():
                    if annotation["visibility"] == "not shown":
                        value = -1

                if half == 1:
                    frame = min(frame, feat_half1.shape[0]-1)
                    label_half1[frame][label] = value

                if half == 2:
                    frame = min(frame, feat_half2.shape[0]-1)
                    label_half2[frame][label] = value

        
            

        feat_half1 = feats2clip(torch.from_numpy(feat_half1), 
                        stride=1, off=int(self.window_size_frame/2), 
                        clip_length=self.window_size_frame)

        feat_half2 = feats2clip(torch.from_numpy(feat_half2), 
                        stride=1, off=int(self.window_size_frame/2), 
                        clip_length=self.window_size_frame)

        
        return self.listGames[index], feat_half1, feat_half2, label_half1, label_half2

    def __len__(self):
        return len(self.listGames)

