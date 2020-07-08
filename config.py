# -*- coding: utf-8 -*-

configs_SC_Single_Speaker_ivectors = {
    "input_dim": 40,
    "average_pooling": 301,
    "cnn_configs": [
        [2, 64, 3, 1],
        [64, 64, 3, 1],
        [64, 128, 3, (2, 1)],
        [128, 128, 3, 1]
    ],
    "speaker_embedding_dim": 100,
    "splice_size": 20 * 128 + 100,
    "Linear_dim": 384,
    "Shared_BLSTM_dim": 896,
    "Linear_Shared_layer1_dim": 160,
    "Linear_Shared_layer2_dim": 160,
    "BLSTM_dim": 896,
    "BLSTM_Projection_dim": 160,
    "output_dim": 3,
    "output_speaker": 1
}
