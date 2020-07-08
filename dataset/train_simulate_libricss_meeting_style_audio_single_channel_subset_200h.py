# -*- coding: utf-8 -*-

from reader import LibriSpeech_Force_Alignment_Label_Generate, Single_Channel_Single_Speaker_Unsegment_Feature_Loader_Split_Mixup


mixspec_json = "/home/mkhe/ts-vad/train_data/fbank/SimLibriCSS_train_subset_200h/mixspec.json"
alignment_label_path = "/home/mkhe/ts-vad/train_data/train-force-alignment"
label = LibriSpeech_Force_Alignment_Label_Generate(alignment_label_path, mixspec_json, wav_dir="/home/mkhe/178_home_mkhe/jsalt2020_simulate/simulate_data/data/SimLibriCSS-train/wav", differ_silence_inference_speech=True)

feature_scp = "/home/mkhe/ts-vad/train_data/fbank/SimLibriCSS_train_subset_200h/SimLibriCSS_train_subset_200h_htk.list"
speaker_embedding_txt = "/home/mkhe/ts-vad/i-vector/ivectors_librispeech_all_cmn_segmented_w3.0_p1.5/cmvn.txt"
single_speaker_3classes_without_mixup = Single_Channel_Single_Speaker_Unsegment_Feature_Loader_Split_Mixup(feature_scp, speaker_embedding_txt, label, max_utt_durance=800, frame_shift=600, mixup_rate=-1, alpha=0.5) # silence, target speaker, inference speaker
