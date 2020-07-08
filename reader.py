# -*- coding: utf-8 -*-

import torch
import numpy as np
import os
import math
import scipy.io as sio
import json
import copy
import HTK


class LoadSpeakerEmbedding():
    def __init__(self, speaker_embedding_txt, cuda=True):
        self.speaker_embedding = self.load_ivector(speaker_embedding_txt, cuda)

    def load_ivector(self, speaker_embedding_txt, cuda=True):
        SCP_IO = open(speaker_embedding_txt)
        speaker_embedding = {}
        raw_lines = [l for l in SCP_IO]
        SCP_IO.close()
        for i in range(len(raw_lines) // 3):
            speaker = raw_lines[3*i].split()[0]
            mean_ivector = np.array(raw_lines[3*i+1].split(), np.float32)
            mean_ivector = mean_ivector[:-1] / mean_ivector[-1]
            if cuda:
                speaker_embedding[speaker] = torch.from_numpy(mean_ivector).cuda()
            else:
                speaker_embedding[speaker] = torch.from_numpy(mean_ivector)
        return speaker_embedding

    def get_speaker_embedding(self, speaker):
        if not speaker in self.speaker_embedding.keys():
            print("{} not in sepaker embedding list".format(speaker))
            exit()
        return self.speaker_embedding[speaker]


def collate_fn_multi_channel(batch, shuffle=True):
    length = [item[2].shape[1] for item in batch]
    ordered_index = sorted(range(len(length)), key=lambda k: length[k] , reverse = True)
    #print(ordered_index)
    nframes = []
    input_data = []
    speaker_embedding = []
    label_data = []
    speaker_index = np.array(range(4))
    channel, Time, Freq = batch[ordered_index[0]][0].shape
    batch_size = len(length)
    input_data = np.zeros([batch_size, channel, Time, Freq]).astype(np.float32)
    for i, id in enumerate(ordered_index):
        if shuffle:
            np.random.shuffle(speaker_index)
        input_data[i, :, :length[id], :] = batch[id][0]
        speaker_embedding.append(batch[id][1][speaker_index])
        label_data.append(torch.from_numpy(batch[id][2][speaker_index]))    # nspeaker * T * 2
        nframes.append(length[id])
    input_data = torch.from_numpy(input_data).transpose(2, 3)
    speaker_embedding = torch.stack(speaker_embedding)
    label_data = torch.cat(label_data, dim=1)  # nspeaker * (Time_Batch1 + Time_Batch2 + ... + Time_BatchN) * 2
    return input_data, speaker_embedding , label_data, nframes


def collate_fn_single_channel(batch):
    '''
    batch: B * (data, embedding, label)
    '''
    num_speaker = batch[0][1].shape[0]
    length = [item[2].shape[1] for item in batch]
    ordered_index = sorted(range(len(length)), key=lambda k: length[k], reverse = True)
    #print(ordered_index)
    nframes = []
    input_data = []
    speaker_embedding = []
    label_data = []
    speaker_index = np.array(range(num_speaker))
    Time, Freq = batch[ordered_index[0]][0].shape
    batch_size = len(length)
    input_data = np.zeros([batch_size, Time, Freq]).astype(np.float32)
    for i, id in enumerate(ordered_index):
        np.random.shuffle(speaker_index)
        input_data[i, :length[id], :] = batch[id][0]
        speaker_embedding.append(batch[id][1][speaker_index])
        label_data.append(torch.from_numpy(batch[id][2][speaker_index].astype(np.long)))
        nframes.append(length[id])
    input_data = torch.from_numpy(input_data).transpose(1, 2) # B * T * F => B * F * T
    speaker_embedding = torch.stack(speaker_embedding) # B * Speaker * Embedding_dim
    label_data = torch.cat(label_data, dim=1)  # nspeaker * (Time_Batch1 + Time_Batch2 + ... + Time_BatchN)
    #print(torch.sum(label_data))
    return input_data, speaker_embedding, label_data, nframes


class Multi_Channel_Unsegment_Feature_Loader_Split_Mixup():
    def __init__(self, feature_scp, speaker_embedding_scp, label, max_utt_durance=800, frame_shift=None, mixup_rate=0, alpha=0.5, num_channel=10):
        self.max_utt_durance = max_utt_durance
        if frame_shift == None:
            self.frame_shift = self.max_utt_durance // 2
        else:
            self.frame_shift = frame_shift
        self.label = label
        self.mixup_rate = mixup_rate
        self.alpha = alpha
        self.num_channel = num_channel
        self.feature_list = self.get_feature_info(feature_scp)
        self.speaker_to_feature_list = self.session_to_feature(self.feature_list) 
        self.speaker_embedding = LoadSpeakerEmbedding(speaker_embedding_scp, cuda=False)

    def get_feature_info(self, feature_scp):
        feature_list = []
        file_list = {}
        min_durance_list = {}
        with open(feature_scp) as SCP_IO:
            for l in SCP_IO:
                '''
                basename: S03_U06.CH2.fea
                '''
                wav_name = os.path.basename(l)
                session = wav_name.split('_')[0]
                channel = "{}_{}".format(wav_name.split('.')[0].split('_')[1], wav_name.split('.')[1]) 
                durance = HTK.readHtk_info(l.rstrip())[0]
                if session not in file_list.keys():
                    file_list[session] = {}
                min_durance_list[session] = 4 * 3600 * 100
                if durance < min_durance_list[session]: min_durance_list[session] = durance
                file_list[session][channel] = l.rstrip()
        for session in file_list.keys():
            if len(file_list[session].keys()) < self.num_channel:
                continue
            start, end = 0, min_durance_list[session]
            total_frame = end - start - 2
            cur_frame = 0
            while(cur_frame < total_frame):
                if cur_frame + self.max_utt_durance <= total_frame:
                    utt_path = []
                    for channel in file_list[session].keys():
                        utt_path.append(file_list[session][channel])
                        feature_list.append((utt_path, session, start, cur_frame, cur_frame+self.max_utt_durance))
                        cur_frame += self.frame_shift
                else:
                    utt_path = []
                    for channel in file_list[session].keys():
                        utt_path.append(file_list[session][channel])
                        cur_frame = max(0, total_frame-self.max_utt_durance)
                        feature_list.append((utt_path, session, start, cur_frame, total_frame)) 
                    break
        return feature_list

    def session_to_feature(self, feature_list):
        speaker_to_feature_list = {}
        for l in feature_list:
            session = l[1]
            if session not in speaker_to_feature_list.keys():
                speaker_to_feature_list[session] = []
            speaker_to_feature_list[session].append(l)
        return speaker_to_feature_list

    def load_fea(self, path, start, end):
        nSamples, sampPeriod, sampSize, parmKind, data = HTK.readHtk_start_end(path, start, end)
        htkdata= np.array(data).reshape(end - start, int(sampSize / 4))
        return end - start, htkdata

    def __len__(self):
        return len(self.feature_list)
    
    def __getitem__(self, idx):
        l = self.feature_list[idx]
        path, session, abs_start, start, end = l
        # load mfcc feature (Channel * T * F)
        channel_inx = np.array(range(len(path)))
        np.random.shuffle(channel_inx)
        mutli_channel_data = []
        for ch in channel_inx[:self.num_channel]:
            total_frame, data = self.load_fea(path[ch], start, end)
            mutli_channel_data.append(data)
        mutli_channel_data = np.vstack(mutli_channel_data).reshape(self.num_channel, total_frame, -1)
        # load label (Speaker * T)
        mask_label, speaker_list = self.label.get_utterance_label(session, None, None, abs_start+start, abs_start+end)
        # load embedding (Speaker * Embedding_dim)
        if np.random.uniform() <= self.mixup_rate:
            #print(len(self.speaker_to_feature_list[session]))
            path, session, abs_start, start, end = self.speaker_to_feature_list[session][np.random.choice(range(len(self.speaker_to_feature_list[session])))]
            channel_inx = np.array(range(len(path)))
            np.random.shuffle(channel_inx)
            mutli_channel_data_2 = []
            for ch in channel_inx[:self.num_channel]:
                total_frame, data = self.load_fea(path[ch], start, end)
                mutli_channel_data_2.append(data)
            mutli_channel_data_2 = np.vstack(mutli_channel_data_2).reshape(self.num_channel, total_frame, -1)
            # load label (Speaker * T)
            mask_label_2, speaker_list_2 = self.label.get_utterance_label(session, None, None, abs_start+start, abs_start+end)
            if speaker_list != speaker_list_2:
                print("speaker order not same")
            else:
                weight = np.random.beta(self.alpha, self.alpha)
                mutli_channel_data = weight * mutli_channel_data + (1 - weight) * mutli_channel_data_2
                mask_label = weight * mask_label + (1 - weight) * mask_label_2
        speaker_embedding = []

        for spk in speaker_list:
            speaker_embedding.append(self.speaker_embedding.get_speaker_embedding(spk)) 
        speaker_embedding = torch.stack(speaker_embedding)
        return mutli_channel_data, speaker_embedding, mask_label


class Single_Channel_Single_Speaker_Unsegment_Feature_Loader_Split_Mixup():
    def __init__(self, feature_scp, speaker_embedding_txt, label, max_utt_durance=800, frame_shift=None, mixup_rate=0, alpha=0.5):
        self.max_utt_durance = max_utt_durance
        if frame_shift == None:
            self.frame_shift = self.max_utt_durance // 2
        else:
            self.frame_shift = frame_shift
        self.label = label
        self.mixup_rate = mixup_rate #mixup_rate<0 means not perform mixup strategy when training
        self.alpha = alpha
        self.feature_list = self.get_feature_info(feature_scp)
        self.speaker_to_feature_list = self.speaker_to_feature(self.feature_list) 
        self.speaker_embedding = LoadSpeakerEmbedding(speaker_embedding_txt, cuda=False)

    def get_feature_info(self, feature_scp):
        feature_list = []
        file_list = {}
        durance_list = {}
        with open(feature_scp) as SCP_IO:
            for l in SCP_IO:
                '''
                basename: 0000.fea
                '''
                session = os.path.basename(l).split('.')[0]
                cur_frame = 0
                total_frame = HTK.readHtk_info(l.rstrip())[0]
                while(cur_frame < total_frame):
                    if cur_frame + self.max_utt_durance <= total_frame:
                        for speaker in self.label.mixspec[session]["speakers"]:
                            feature_list.append((l.rstrip(), "{}_{}".format(session, speaker), cur_frame, cur_frame+self.max_utt_durance))
                        cur_frame += self.frame_shift
                    else:
                        cur_frame = max(0, total_frame-self.max_utt_durance)
                        for speaker in self.label.mixspec[session]["speakers"]:
                            feature_list.append((l.rstrip(), "{}_{}".format(session, speaker), cur_frame, total_frame))
                        break
        return feature_list

    def speaker_to_feature(self, feature_list):
        speaker_to_feature_list = {}
        for l in feature_list:
            session, speaker = l[1].split('_')
            if speaker not in speaker_to_feature_list.keys():
                speaker_to_feature_list[speaker] = []
            speaker_to_feature_list[speaker].append(l)
        return speaker_to_feature_list

    def load_fea(self, path, start, end):
        nSamples, sampPeriod, sampSize, parmKind, data = HTK.readHtk_start_end(path, start, end)
        htkdata= np.array(data).reshape(end - start, int(sampSize / 4))
        return end - start, htkdata

    def __len__(self):
        return len(self.feature_list)
    
    def __getitem__(self, idx):
        l = self.feature_list[idx]
        path, session_speaker, start, end = l
        session, speaker = session_speaker.split('_')
        # load feature (T * F)
        total_frame, data = self.load_fea(path, start, end)
        # load label (Speaker * T)
        mask_label = self.label.get_mixture_utternce_label(session, target_speaker=speaker, start=start, end=end)
        # load embedding (Speaker * Embedding_dim)
        if np.random.uniform() <= self.mixup_rate:
            #print(len(self.speaker_to_feature_list[session]))
            path, session_speaker, start, end = self.speaker_to_feature_list[session][np.random.choice(range(len(self.speaker_to_feature_list[speaker])))]
            session, speaker = session_speaker.split('_')
            total_frame, data_2 = self.load_fea(path[ch], start, end)
            mask_label_2 = self.label.get_mixture_utternce_label(session, target_speaker=speaker, start=start, end=end)
            weight = np.random.beta(self.alpha, self.alpha)
            data = weight * data + (1 - weight) * data_2
            mask_label = weight * mask_label + (1 - weight) * mask_label_2
        speaker_embedding = self.speaker_embedding.get_speaker_embedding(speaker)
        '''
        returns:
        data: [T, F]
        speaker_embedding: [num_speaker, embedding_dim]
        mask_label: [num_speaker, T]
        '''
        return data, speaker_embedding[None, :], mask_label[None, :]


class LibriSpeech_Force_Alignment_Label_Generate():
    def __init__(self, alignment_label_path, mixspec_json, wav_dir=None, differ_silence_inference_speech=False):
        '''
        103-1240-0000 1 0.430 0.400 CHAPTER 
        103-1240-0000 1 0.830 0.330 ONE 
        103-1240-0000 1 1.450 0.380 MISSUS 
        103-1240-0000 1 1.830 0.350 RACHEL 
        103-1240-0000 1 2.180 0.350 LYNDE 
        103-1240-0000 1 2.530 0.190 IS
        '''
        self.alignment_label_path = alignment_label_path
        self.mixspec = self.init_mixspec(mixspec_json)
        self.wav_dir = wav_dir
        self.source_utterance_to_force_alignment_array = {}
        self.mixture_utterance_to_force_alignment_array = {}
        # if differ_silence_inference_speech == True then the function will return 3 class: silence, target speech, inference speech
        self.differ_silence_inference_speech = differ_silence_inference_speech

    
    def init_mixspec(self, mixspec_json):
        mixspec = {}
        with open(mixspec_json, 'r', encoding='utf-8') as INPUT:
            for i in json.load(INPUT):
                #print(i.keys())
                mixture_id = os.path.basename(i["output"]).split('.')[0]
                mixspec[mixture_id] = {}
                mixspec[mixture_id]["inputs"] = i["inputs"]
                mixspec[mixture_id]["output"] = i["output"]
                mixspec[mixture_id]["speakers"] = i["speakers"]
        return mixspec

    def get_source_utterance_label(self, utterance, start=0, end=None):
        '''
        utterance = "103-1240-0000"
        '''
        spk, story, uid = utterance.split('-')
        '''
        if spk not in self.source_utterance_to_force_alignment_array.keys():
            self.source_utterance_to_force_alignment_array[spk] = {}
        if story not in self.source_utterance_to_force_alignment_array[spk].keys():
            self.source_utterance_to_force_alignment_array[spk][story] = {}
        if uid not in self.source_utterance_to_force_alignment_array[spk][story].keys():
            self.source_utterance_to_force_alignment_array[spk][story][uid] = np.load("{}/{}/{}/{}.npy".format(self.alignment_label_path, spk, story, utterance))
        return self.source_utterance_to_force_alignment_array[spk][story][uid][start:end]
        '''
        return np.load("{}/{}/{}/{}.npy".format(self.alignment_label_path, spk, story, utterance))[start:end]

    def get_mixture_utternce_label(self, mixture_id, target_speaker=None, start=0, end=None):
        if mixture_id not in self.mixture_utterance_to_force_alignment_array.keys():
            self.mixture_utterance_to_force_alignment_array[mixture_id] = {}
            if self.wav_dir != None:
                durance = int(math.ceil(int(os.popen('soxi {}/{}.wav | grep Duration| cut -d "=" -f 2 |cut -d "s" -f 1'.format(self.wav_dir, mixture_id)).readlines()[0]) / 160))
            else:
                durance = 10 * 60 * 100
            for speaker in self.mixspec[mixture_id]["speakers"]:
                self.mixture_utterance_to_force_alignment_array[mixture_id][speaker] = np.zeros(durance, dtype=np.int8)
            for source_utterance in self.mixspec[mixture_id]["inputs"]:
                source_utterance_label = self.get_source_utterance_label(source_utterance["utterance_id"])
                offset = int(100 * source_utterance["offset"])
                offset_durance = offset + len(source_utterance_label)
                #print(source_utterance_label.shape)
                #print("{} {}".format(offset, offset_durance))
                if offset_durance > durance:
                    print("offset{} offset_durance{} source_utterance_label{} durance{} mixture_id{}".format(offset, offset_durance, len(source_utterance_label), durance, mixture_id))
                self.mixture_utterance_to_force_alignment_array[mixture_id][speaker][offset:offset_durance] = source_utterance_label
            if self.differ_silence_inference_speech:
                num_speaker = 0
                temp_label = {}
                for speaker in self.mixture_utterance_to_force_alignment_array[mixture_id]:
                    num_speaker += self.mixture_utterance_to_force_alignment_array[mixture_id][speaker]
                for speaker in self.mixture_utterance_to_force_alignment_array[mixture_id]:
                    num_inference_speaker = num_speaker - self.mixture_utterance_to_force_alignment_array[mixture_id][speaker]
                    temp_label[speaker] = copy.deepcopy(self.mixture_utterance_to_force_alignment_array[mixture_id][speaker])
                    without_target_speaker_mask = self.mixture_utterance_to_force_alignment_array[mixture_id][speaker] == 0
                    # 3 class: silence(0), target speech(1), inference speech(2)
                    temp_label[speaker][without_target_speaker_mask & (num_inference_speaker>0)] = 2
                self.mixture_utterance_to_force_alignment_array[mixture_id] = temp_label
        mixture_utternce_label = {}
        if target_speaker != None:
            return self.mixture_utterance_to_force_alignment_array[mixture_id][target_speaker][start:end]
        else:
            for speaker in self.mixture_utterance_to_force_alignment_array[mixture_id].keys():
                mixture_utternce_label[speaker] = self.mixture_utterance_to_force_alignment_array[mixture_id][speaker][start:end]
            return mixture_utternce_label

