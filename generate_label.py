# -*- coding: utf-8 -*-

import os

import numpy as np
from tqdm import tqdm

from reader import LibriSpeech_Force_Alignment_Label_Generate

mixspec_json = "/home/mkhe/ts-vad/train_data/fbank/SimLibriCSS_train_subset_200h/mixspec.json"
alignment_label_path = "/home/mkhe/ts-vad/train_data/train-force-alignment"
label = LibriSpeech_Force_Alignment_Label_Generate(alignment_label_path, mixspec_json,
                                                   wav_dir="/home/mkhe/178_home_mkhe/jsalt2020_simulate/simulate_data/data/SimLibriCSS-train/wav",
                                                   differ_silence_inference_speech=True)
feature_scp = "/home/mkhe/ts-vad/train_data/fbank/SimLibriCSS_train_subset_200h/SimLibriCSS_train_subset_200h_htk.list"
output_dir = "train_data/label/SimLibriCSS_train_subset_200h"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
with open(feature_scp) as INPUT:
    mixture_list = [l for l in INPUT]
    for l in tqdm(mixture_list):
        mixture_id = os.path.basename(l).split('.')[0]
        mixture_label = label.get_mixture_utternce_label(mixture_id)
        for speaker in mixture_label.keys():
            np.save(os.path.join(output_dir, "{}_{}.npy".format(mixture_id, speaker)), mixture_label[speaker])
