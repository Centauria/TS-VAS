# -*- coding: utf-8 -*-

import os
import numpy as np


root_dir = "/home/mkhe/libricss/libri_css/exp/data-orig/for_release/OV10/overlap_ratio_10.0_sil0.1_1.0_session1_actual10.2/transcription"
transcription = []
for d1 in os.listdir(root_dir):
    for d2 in os.listdir(os.path.join(root_dir, d1)):
        transcription.append((d2, os.path.join(root_dir, d1, d2, "transcription/meeting_info.txt")))

'''
start_time	end_time	speaker	utterance_id	transcription
3.000000	7.290000	1995	1995-1837-0015	THE SQUARES OF COTTON SHARP EDGED HEAVY WERE JUST ABOUT TO BURST TO BOLLS
6.425313	21.835313	8463	8463-287645-0005	A FEW YEARS BACK ONE OF THEIR SLAVES A COACHMAN WAS KEPT ON THE COACH BOX ONE COLD NIGHT WHEN THEY WERE OUT AT A BALL UNTIL HE BECAME ALMOST FROZEN TO DEATH IN FACT HE DID DIE IN THE INFIRMARY FROM THE EFFECTS OF THE FROST ABOUT ONE WEEK AFTERWARDS
22.583250	24.573250	8455	8455-210777-0054	THE LETTER RAN AS FOLLOWS
'''
rttm = {}
for tr in transcription:
    session = tr[0]
    rttm[session] = {}
    with open(transcription[1]) as INPUT:
        lines = [l for l in INPUT]
        for l in lines[1:]:
            start_time, end_time, speaker = l.split('\t')[:3]
            start_time = int(float(start_time) * 100)
            end_time = int(float(end_time) * 100)
            if speaker not in rttm[session].keys():
                rttm[session][speaker] = np.zeros(11 * 60 * 100)
            rttm[session][speaker][start_time:end_time] = 1
        rttm[session]["sum"] = 0
        for speaker in rttm[session].keys():
            rttm[session]["sum"] += rttm[session][speaker]

session_num_speaker = {}
overlap_num_speaker = {}

for session in rttm.keys():
    num_speaker = len(rttm[session].keys()) - 1
    if str(num_speaker) not in session_num_speaker.keys():
        session_num_speaker[str(num_speaker)] = 0
    session_num_speaker[str(num_speaker)] += 1
    for i in range(num_speaker+1):
        if str(num_speaker) not in overlap_num_speaker.keys():
            overlap_num_speaker[str(i)] = 0
        overlap_num_speaker[str(i)] += np.sum(rttm[session]["sum"]==i)

print(session_num_speaker)
for num_speaker in overlap_num_speaker.keys():
    print("{}: {}".format(num_speaker, overlap_num_speaker[num_speaker]/100/60))
