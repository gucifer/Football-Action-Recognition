from torch.utils.data import Dataset

import numpy as np
import random
import os
import time
import socket

from tqdm import tqdm
import argparse

import torch

import logging
import json

from SoccerNet.Downloader import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
from .load_data_utils import LABELS


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
    def __init__(self, path, features="ResNET_PCA512.npy", split=["train"], version=1, 
                framerate=2, window_size=15, n_download = 500):
        self.path = path
        self.listGames = getListGames(split)[:n_download]
        self.features = features
        self.window_size_frame = window_size*framerate
        self.version = version
        if version == 1:
            self.num_classes = 3
            self.labels="Labels.json"
        elif version == 2:
            self.dict_event = LABELS
            self.num_classes = 17
            self.labels="Labels-v2.json"

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)
        if ".local" in socket.gethostname():
            downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=split, verbose=False,randomized=False, n_download = n_download)
        else:
            downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=split, verbose=False,randomized=False)


        logging.info("Pre-compute clips")

        self.game_feats = list()
        self.game_labels = list()
        self.rel_offsets = list()
        self.matches = list()
        self.halves = list()
        self.start_frame_indices = list()

        # game_counter = 0
        for game in tqdm(self.listGames):
            # game_counter += 1
            # Load features
            feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features))
            feat_half1 = feat_half1.reshape(-1, feat_half1.shape[-1])
            feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features))
            feat_half2 = feat_half2.reshape(-1, feat_half2.shape[-1])

            feat_half1 = feats2clip(torch.from_numpy(feat_half1), stride=self.window_size_frame, clip_length=self.window_size_frame)
            feat_half2 = feats2clip(torch.from_numpy(feat_half2), stride=self.window_size_frame, clip_length=self.window_size_frame)

            # Load labels
            labels = json.load(open(os.path.join(self.path, game, self.labels)))

            label_half1 = np.zeros((feat_half1.shape[0], self.num_classes+1), dtype=np.float32)
            label_half1[:,-1]=1 # those are BG classes
            label_half2 = np.zeros((feat_half2.shape[0], self.num_classes+1), dtype=np.float32)
            label_half2[:,-1]=1 # those are BG classes

            rel_offset1 = np.zeros((feat_half1.shape[0]), dtype=np.float32)
            rel_offset2 = np.zeros((feat_half2.shape[0]), dtype=np.float32)


            for annotation in labels["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                frame = framerate * (seconds + 60 * minutes)

                if version == 1:
                    if "card" in event: label = 0
                    elif "subs" in event: label = 1
                    elif "soccer" in event: label = 2
                    else: continue
                elif version == 2:
                    if event not in self.dict_event:
                        continue
                    label = self.dict_event[event]

                temporal_viewpoint = frame//self.window_size_frame
                rel_offset = (frame - temporal_viewpoint * self.window_size_frame)/self.window_size_frame

                # if label outside temporal of view
                if half == 1 and temporal_viewpoint>=label_half1.shape[0]:
                    continue
                if half == 2 and temporal_viewpoint>=label_half2.shape[0]:
                    continue
                
                if half == 1:
                    label_half1[temporal_viewpoint][-1] = 0 # not BG anymore
                    label_half1[temporal_viewpoint][label] = 1 # that's my class
                    rel_offset1[temporal_viewpoint] = rel_offset

                if half == 2:
                    label_half2[temporal_viewpoint][-1] = 0 # not BG anymore
                    label_half2[temporal_viewpoint][label] = 1 # that's my class
                    rel_offset2[temporal_viewpoint] = rel_offset
            
            self.game_feats.append(feat_half1)
            self.game_feats.append(feat_half2)
            self.game_labels.append(label_half1)
            self.game_labels.append(label_half2)
            self.rel_offsets.append(rel_offset1)
            self.rel_offsets.append(rel_offset2)
            self.matches.append([game] * (feat_half1.shape[0] + feat_half2.shape[0]))
            self.halves.append([1] * feat_half1.shape[0] +  [2] * feat_half2.shape[0])
            self.start_frame_indices.append(np.concatenate((np.arange(0, feat_half1.shape[0]) * 40,  np.arange(0, feat_half2.shape[0]) * 40)))

        self.game_feats = np.concatenate(self.game_feats)
        self.game_labels = np.concatenate(self.game_labels)
        self.rel_offsets = np.concatenate(self.rel_offsets)
        self.matches = np.concatenate(self.matches)
        self.halves = np.concatenate(self.halves)
        self.start_frame_indices = np.concatenate(self.start_frame_indices)
        halt=1



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            clip_feat (np.array): clip of features.
            clip_labels (np.array): clip of labels for the segmentation.
            clip_targets (np.array): clip of targets for the spotting.
        """
        return self.game_feats[index,:,:], self.game_labels[index,:], self.rel_offsets[index], self.matches[index], self.halves[index], self.start_frame_indices[index]

    def __len__(self):
        return len(self.game_feats)
