# -*- coding: utf-8 -*-

import os
import sys
from train import Train_SingleGPU
from reader import collate_fn_single_channel
from model import TS_VAD_SC
import config
import torch
from loss_function import CrossEntropy_SingleTargets

from dataset.train_simulate_libricss_meeting_style_audio_single_channel_subset_200h import single_speaker_3classes_without_mixup as single_speaker_3classes_without_mixup

output_dir = "model/TS_VAD_SC/Batchsize32_Subset_200h_Channel1_Segment8s_lr0.05_configs_SC_Single_Speaker_ivectors_ForceAlignmentLabel_WithoutMixup"
if not os.path.exists (output_dir):
    os.makedirs(output_dir)
os.system("cp {} {}/{}".format(os.path.abspath(sys.argv[0]), output_dir, sys.argv[0]))
optimizer = torch.optim.Adam

train = Train_SingleGPU(single_speaker_3classes_without_mixup, collate_fn_single_channel, TS_VAD_SC, config.configs_SC_Single_Speaker_ivectors, "TS_VAD_MC", output_dir, optimizer, CrossEntropy_SingleTargets, batchsize=32, accumulation_steps=[(0, 1)], lr=0.05, start_epoch=0, end_epoch=10, cuda=2, num_workers=4)

train.train()