# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np


class LSTM_Projection(nn.Module):
    def __init__(self, input_size, hidden_size, linear_dim, num_layers=1, bidirectional=True, dropout=0):
        super(LSTM_Projection, self).__init__()
        self.hidden_size = hidden_size
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)
        self.forward_projection = nn.Linear(hidden_size, linear_dim)
        self.backward_projection = nn.Linear(hidden_size, linear_dim)
        self.relu = nn.ReLU(True)

    def forward(self, x, nframes):
        '''
        x: [batchsize, Time, Freq]
        nframes: [len_b1, len_b2, ..., len_bN]
        '''
        packed_x = nn.utils.rnn.pack_padded_sequence(x, nframes, batch_first=True)
        packed_x_1, hidden = self.LSTM(packed_x)
        x_1, l = nn.utils.rnn.pad_packed_sequence(packed_x_1, batch_first=True)
        forward_projection = self.relu(self.forward_projection(x_1[..., :self.hidden_size]))
        backward_projection = self.relu(self.backward_projection(x_1[..., self.hidden_size:]))
        # x_2: [batchsize, Time, linear_dim*2]
        x_2 = torch.cat((forward_projection, backward_projection), dim=2)
        return x_2


class LSTM_Projection_Multi_Layer(nn.Module):
    def __init__(self, configs, bidirectional=True, dropout=0):
        super(LSTM_Projection_Multi_Layer, self).__init__()
        self.num_layers = len(configs.keys()) - 1
        self.lstm_projection_multi_layer = nn.Sequential()
        for l in range(self.num_layers):
            self.lstm_projection_multi_layer.add_module("LSTM_Projection{}".format(l), LSTM_Projection(configs[str(l)][0], configs[str(l)][1], configs[str(l)][2], bidirectional=bidirectional, dropout=dropout))
        self.FC = nn.Linear(configs["FC"][0], configs["FC"][1])
    
    def forward(self, x, nframes):
        x_1 = self.lstm_projection_multi_layer(x, nframes)
        x_2 = self.FC(x_1)
        return x_2


class CNN2D_BN_Relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(CNN2D_BN_Relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels) #(N,C,H,W) on C
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, bias=True):
        super(SeparableConv1d,self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size, stride, kernel_size//2, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels) #(N,C,L) on C
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TS_VAD_MC(nn.Module):
    '''
    INPUT (MFCC)
    IDCT: MFCC to FBANK
    Batchnorm
    Stats pooling: batchnorm-cmn
    Combine inputs:  (batchnorm, batchnorm-cmn Speaker Detection Block
    4 layers CNN: Conv2d (in channels, out channels, kernel size, stride=1, padding=0 dilation=1, groups=1, bias=True, padding mode=zeros) 1 layer Splice-embedding:  (Convld SD, ivector-k) Linear layer
    2 layers Shared-blstm
    Attention
    1 layers CNN: Convld (in channels, out channels, kernel size, stride=1, padding=0 dilation=1, groups=1, bias=True, padding mode=zeros) Attention layer
    1 layers Combine-speaker
    1 layers BLSTM
    1 layers Dense: 4 dependent FC layers
    '''
    def __init__(self, configs):
        super(TS_VAD_MC, self).__init__()
        self.input_size = configs["input_dim"]
        self.cnn_configs = configs["cnn_configs"]
        self.speaker_embedding_size = configs["speaker_embedding_dim"]
        self.Linear_dim = configs["Linear_dim"]
        self.Shared_BLSTM_size = configs["Shared_BLSTM_dim"]
        self.Linear_Shared_layer1_dim = configs["Linear_Shared_layer1_dim"]
        self.Linear_Shared_layer2_dim = configs["Linear_Shared_layer2_dim"] 
        self.cnn_attention = configs["cnn_attention"]
        #self.attention_hidden_dim = configs["attention_hidden_dim"]
        self.BLSTM_size = configs["BLSTM_dim"]
        self.BLSTM_Projection_dim = configs["BLSTM_Projection_dim"]
        self.output_size = configs["output_dim"]
        self.output_speaker = configs["output_speaker"]

        self.idct = torch.from_numpy(np.load('./dataset/idct.npy').astype(np.float32)).cuda() 
        self.batchnorm = nn.BatchNorm2d(1)
        self.average_pooling = nn.AvgPool1d(configs["average_pooling"], stride=1, padding=configs["average_pooling"]//2)
        # Speaker Detection Block
        self.Conv2d_SD = nn.Sequential()
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD1', CNN2D_BN_Relu(self.cnn_configs[0][0], self.cnn_configs[0][1], self.cnn_configs[0][2], self.cnn_configs[0][3]))
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD2', CNN2D_BN_Relu(self.cnn_configs[1][0], self.cnn_configs[1][1], self.cnn_configs[1][2], self.cnn_configs[1][3]))
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD3', CNN2D_BN_Relu(self.cnn_configs[2][0], self.cnn_configs[2][1], self.cnn_configs[2][2], self.cnn_configs[2][3]))
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD4', CNN2D_BN_Relu(self.cnn_configs[3][0], self.cnn_configs[3][1], self.cnn_configs[3][2], self.cnn_configs[3][3]))

        self.splice_size = configs["splice_size"]
        self.Linear = nn.Linear(self.splice_size, self.Linear_dim)
        self.relu = nn.ReLU(True)
        self.Shared_BLSTMP_1 = LSTM_Projection(input_size=self.Linear_dim, hidden_size=self.Shared_BLSTM_size, linear_dim=self.Linear_Shared_layer1_dim, num_layers=1, bidirectional=True, dropout=0)
        self.Shared_BLSTMP_2 = LSTM_Projection(input_size=self.Linear_Shared_layer1_dim*2, hidden_size=self.Shared_BLSTM_size, linear_dim=self.Linear_Shared_layer2_dim, num_layers=1, bidirectional=True, dropout=0)
        self.Conv1d_Attention = nn.Sequential()
        if self.Linear_Shared_layer2_dim*2 != self.cnn_attention[0]:
            print("input dim doesn't match to input channel number")
            exit()
        self.Conv1d_Attention.add_module('Conv1d_Attention', SeparableConv1d(self.cnn_attention[0], self.cnn_attention[1], self.cnn_attention[2], self.cnn_attention[3])) 
        #self.Attention = SelfAttention(self.cnn_attention[1], self.attention_hidden_dim)

        self.conbine_speaker_size = self.cnn_attention[1] * 4
        self.BLSTMP = LSTM_Projection(input_size=self.conbine_speaker_size, hidden_size=self.BLSTM_size, linear_dim=self.BLSTM_Projection_dim, num_layers=1, bidirectional=True, dropout=0)
        self.FC = {}
        for i in range(self.output_speaker):
            self.FC[str(i)] = nn.Linear(self.BLSTM_Projection_dim*2, self.output_size)

    def forward(self, x, embedding, nframes):
        '''
        x: Batch * Channel * Freq * Time
        embedding :
        Batch * speaker(4) * Embedding
        nframe: descend order
        '''
        batchsize, Channel, Freq, Time = x.shape
        speaker = embedding.shape[1]
        embedding_dim = self.speaker_embedding_size
        # ****************IDCT: MFCC to FBANK*****************
        # [batchsize, Channel, Freq, Time] -> [Freq, Channel, batchsize, Time] -> [Freq, -1] 
        x_1 = x.transpose(0, 2).reshape(Freq, -1)
        # MFCC to FBANK
        x_2 = self.idct.mm(x_1).reshape(Freq, Channel, batchsize, Time).transpose(0, 2)

        # ************batchnorm statspooling*****************
        #batchnorm [batchsize, Channel, Freq, Time] -> [ batchsize*Channel, 1, Freq, Time]
        x_3 = self.batchnorm(x_2.reshape(batchsize*Channel, 1, Freq, Time)).squeeze(dim=1)
        #batchnorm_cmn
        #w = Time / torch.Tensor (nframes)
        #x_3_mean = torch.mean(x_3, dim=3) * w[:, None, None ]
        x_3_mean = self.average_pooling(x_3)
        #(batchnorm, batchnorm_cmn): 2 * [batchsize*Channel, Freq, Time] -> [batchs ize*Channel, 2, Freq, Time]
        #x_4= torch.cat(x_3, x_3_mean [ None].repeat(1, 1, 1, Time) , dim=2)
        x_4 = torch.cat((x_3, x_3_mean), dim=1).reshape(batchsize*Channel, 2, Freq, Time)
        #(batchnorm, batchnorm_cmn) -> (bn_d0, bnc_d0, bn_d1, bnc_d1,..., bn_d40, bnc_d40)
        #x_5 = x_4.reshape(batchsize, Channel, 2, Freq, Time).transpose(2, 3).reshape(batchsize, Channel, 2*Freq, Time)

        # **************CNN*************
        # [batchsize*Channel, 2，Freq, Time] -〉[batchsize*Channel, Conv-4-out-filters, Freq, Time]
        x_5 = self.Conv2d_SD(x_4)
        #[batchsize*Channel, Conv-4-out-filters, Freq, Time] -〉 [ batchsize*Channel, Conv-4-out- filters*Freq, Time ]
        x_6 = x_5.reshape(batchsize*Channel, -1, Time)
        Freq = x_6.shape[1]
        #*********************************************need to check***************** ***
        #print(x_1.repeat(1, speaker, 1).shape)
        x_6_reshape = x_6.repeat(1, speaker, 1).reshape(batchsize * Channel * speaker, Freq, Time)
        #print(x_1_reshape.shape)
        embedding_reshape = embedding.repeat(1, Channel, 1).reshape(-1, embedding_dim)[..., None].expand(batchsize * Channel * speaker, embedding_dim, Time)
        #print(embedding_reshape.shape)
        x_7 = torch.cat((x_6_reshape, embedding_reshape), dim=1)
        '''
        x_7: (Batch * Channel * speaker) * (Freq + Embedding) * Time
        1stBatch 1stChannel 1stspeaker (Freq + Embedding) * Time1stBatch 1stChannel 2ndspeaker (Freq + Embedding) * Time
        1stBatch 2ndChannel 1stspeaker (Freq + Embedding) * Time1stBatch 2ndChannel 2ndspeaker (Freq + Embedding) * Time
        '''
        #**************Linear*************
        #(Batch * Channel * speaker) * (Freq + Embedding) * Time -〉(Batch * Channel * speaker) *Time * Linear_dim
        x_8 = self.relu(self.Linear(x_7.transpose(1, 2)))
        #Shared_BLSTMP_1 (Batch * Channel * speaker) * Time * Linear_dim =〉(Batch * Channel *speaker) * Time * (Linear_Shared_layer1_dim * 2)
        lens = [n for n in nframes for j in range(Channel) for i in range(speaker)] 
        x_9 = self.Shared_BLSTMP_1(x_8, lens)
        #Shared_BLSTMP_2 (Batch * Channel * speaker) * Time * (Linear_Shared_layer1_dim * 2) =〉 (Batch * Channel * speaker) * Time * (Linear_Shared_layer1_dim * 2)
        x_10 = self.Shared_BLSTMP_2(x_9, lens)

        #Attention
        #(Batch * Channel * speaker) * Time * (Linear_Shared_layer1_dim * 2) => (Batch * Channel * speaker) * Time * cnn_attention_out_channe 1
        x_11= self.Conv1d_Attention(x_10.transpose(1, 2)).transpose(1, 2)
        #(Batch * Channel * speaker) * Time * cnn_attention_out_channel => Batch * speaker * Time * cnn_attention_out_channel
        #x_5, weights = self.Attention(x_4.reshape(batchsize, Channel, speaker, Time , self.cnn_attention[1]).transpose(1, 3).transpose(1, 2))
        #***** ** ** ********************* check mean attention 水***水***水** ******* ******
        x_12 = torch.mean(x_11.reshape(batchsize, Channel, speaker, Time, self.cnn_attention[1]).transpose(1, 3).transpose(1, 2), -2)
        #Combine- Speaker: Batch * speaker * Time * cnn_attention_out_channel => Batch * Time * (speaker * cnn_attention_out_channel)
        x_13 = x_12.transpose(1, 2).reshape(batchsize, Time, speaker * self.cnn_attention[1])
        '''
        batchsize * Time * (1stspeaker 2ndspeaker 3rdspeaker 4thspeaker)
        '''

        #BLSTM: Batch * Time * (speaker * self.cnn_attention[1]) => Batch * Time * (BLSTM_Projection_dim * 2)
        x_14 = self.BLSTMP(x_13, nframes)

        #Dimension reduction: remove the padding frames; Batch * Time * (BLSTM_Projection_dim * 2) => (Batch * Time) * (BLSTM_Projection_dim * 2)
        lens = [k for i, m in enumerate(nframes) for k in range(i * Time, m + i * Time)]
        x_15 = x_14.reshape(batchsize * Time, -1)[lens, :]
        '''
        1stBatch(1st sentence) length1 * (BLSTM_size * 2)
        2ndBatch(2nd sentence) length2 * (BLSTM_size * 2)
        ...
        '''
        '''
        out1 = F.softmax(self.FC1(x_8))
        out2 = F.softmax(self.FC2(x_8))
        out3 = F.softmax(self.FC3(x_8))
        out4 = F.softmax(self.FC4(x_8))
        '''
        out = []
        for i in self.FC:
            out.append(self.FC[i](x_15))
        return out


class TS_VAD_SC(nn.Module):
    '''
    INPUT (MFCC)
    IDCT: MFCC to FBANK
    Batchnorm
    Stats pooling: batchnorm-cmn
    Combine inputs:  (batchnorm, batchnorm-cmn Speaker Detection Block
    4 layers CNN: Conv2d (in channels, out channels, kernel size, stride=1, padding=0 dilation=1, groups=1, bias=True, padding mode=zeros) 1 layer Splice-embedding:  (Convld SD, ivector-k) Linear layer
    2 layers Shared-blstm
    Attention
    1 layers CNN: Convld (in channels, out channels, kernel size, stride=1, padding=0 dilation=1, groups=1, bias=True, padding mode=zeros) Attention layer
    1 layers Combine-speaker
    1 layers BLSTM
    1 layers Dense: 4 dependent FC layers
    '''
    def __init__(self, configs):
        super(TS_VAD_SC, self).__init__()
        self.input_size = configs["input_dim"]
        self.cnn_configs = configs["cnn_configs"]
        self.speaker_embedding_size = configs["speaker_embedding_dim"]
        self.Linear_dim = configs["Linear_dim"]
        self.Shared_BLSTM_size = configs["Shared_BLSTM_dim"]
        self.Linear_Shared_layer1_dim = configs["Linear_Shared_layer1_dim"]
        self.Linear_Shared_layer2_dim = configs["Linear_Shared_layer2_dim"]
        #self.attention_hidden_dim = configs["attention_hidden_dim"]
        self.BLSTM_size = configs["BLSTM_dim"]
        self.BLSTM_Projection_dim = configs["BLSTM_Projection_dim"]
        self.output_size = configs["output_dim"]
        self.output_speaker = configs["output_speaker"]

        #self.idct = torch.from_numpy(np.load('./dataset/idct.npy').astype(np.float32)).cuda()  #if input is mfcc
        self.batchnorm = nn.BatchNorm2d(1)
        self.average_pooling = nn.AvgPool1d(configs["average_pooling"], stride=1, padding=configs["average_pooling"]//2)
        # Speaker Detection Block
        self.Conv2d_SD = nn.Sequential()
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD1', CNN2D_BN_Relu(self.cnn_configs[0][0], self.cnn_configs[0][1], self.cnn_configs[0][2], self.cnn_configs[0][3]))
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD2', CNN2D_BN_Relu(self.cnn_configs[1][0], self.cnn_configs[1][1], self.cnn_configs[1][2], self.cnn_configs[1][3]))
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD3', CNN2D_BN_Relu(self.cnn_configs[2][0], self.cnn_configs[2][1], self.cnn_configs[2][2], self.cnn_configs[2][3]))
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD4', CNN2D_BN_Relu(self.cnn_configs[3][0], self.cnn_configs[3][1], self.cnn_configs[3][2], self.cnn_configs[3][3]))

        self.splice_size = configs["splice_size"]
        self.Linear = nn.Linear(self.splice_size, self.Linear_dim)
        self.relu = nn.ReLU(True)
        self.Shared_BLSTMP_1 = LSTM_Projection(input_size=self.Linear_dim, hidden_size=self.Shared_BLSTM_size, linear_dim=self.Linear_Shared_layer1_dim, num_layers=1, bidirectional=True, dropout=0)
        self.Shared_BLSTMP_2 = LSTM_Projection(input_size=self.Linear_Shared_layer1_dim*2, hidden_size=self.Shared_BLSTM_size, linear_dim=self.Linear_Shared_layer2_dim, num_layers=1, bidirectional=True, dropout=0)

        self.conbine_speaker_size = self.Linear_Shared_layer2_dim * 2 * self.output_speaker
        self.BLSTMP = LSTM_Projection(input_size=self.conbine_speaker_size, hidden_size=self.BLSTM_size, linear_dim=self.BLSTM_Projection_dim, num_layers=1, bidirectional=True, dropout=0)
        '''
        self.FC = {}
        for i in range(self.output_speaker):
            self.FC[str(i)] = nn.Linear(self.BLSTM_Projection_dim*2, self.output_size)
        '''
        self.FC = nn.Linear(self.BLSTM_Projection_dim*2, self.output_size)

    def forward(self, x, embedding, nframes):
        '''
        x: Batch * Freq * Time
        embedding :
        Batch * speaker(4) * Embedding
        nframe: descend order
        '''
        batchsize, Freq, Time = x.shape
        speaker = embedding.shape[1]
        embedding_dim = self.speaker_embedding_size
        # ****************IDCT: MFCC to FBANK*****************
        # [batchsize, Freq, Time] -> [Freq, batchsize, Time] -> [Freq, -1] 
        #x_1 = x.transpose(0, 1).reshape(Freq, -1)
        # MFCC to FBANK
        #x_2 = self.idct.mm(x_1).reshape(Freq, batchsize, Time).transpose(0, 1)

        # ************batchnorm statspooling*****************
        #batchnorm [batchsize, Freq, Time] -> [ batchsize, 1, Freq, Time]
        x_3 = self.batchnorm(x.reshape(batchsize, 1, Freq, Time)).squeeze(dim=1)
        #batchnorm_cmn
        #w = Time / torch.Tensor (nframes)
        #x_3_mean = torch.mean(x_3, dim=3) * w[:, None, None ]
        x_3_mean = self.average_pooling(x_3)
        #(batchnorm, batchnorm_cmn): 2 * [batchsize, Freq, Time] -> [batchs ize, 2, Freq, Time]
        x_4 = torch.cat((x_3, x_3_mean), dim=1).reshape(batchsize, 2, Freq, Time)
        #(batchnorm, batchnorm_cmn) -> (bn_d0, bnc_d0, bn_d1, bnc_d1,..., bn_d40, bnc_d40)

        # **************CNN*************
        # [batchsize, 2，Freq, Time] -〉[batchsize, Conv-4-out-filters, Freq, Time]
        x_5 = self.Conv2d_SD(x_4)
        #[batchsize, Conv-4-out-filters, Freq, Time] -〉 [ batchsize, Conv-4-out-filters*Freq, Time ]
        x_6 = x_5.reshape(batchsize, -1, Time)
        Freq = x_6.shape[1]
        #*********************************************need to check***************** ***
        #print(x_1.repeat(1, speaker, 1).shape)
        x_6_reshape = x_6.repeat(1, speaker, 1).reshape(batchsize * speaker, Freq, Time)
        #embedding: Batch * speaker * Embedding -> (Batch * speaker) * Embedding * Time
        embedding_reshape = embedding.reshape(-1, embedding_dim)[..., None].expand(batchsize * speaker, embedding_dim, Time)
        #print(embedding_reshape.shape)
        x_7 = torch.cat((x_6_reshape, embedding_reshape), dim=1)
        '''
        x_7: (Batch * speaker) * (Freq + Embedding) * Time
        '''
        #**************Linear*************
        #(Batch * speaker) * (Freq + Embedding) * Time -〉(Batch * speaker) * Time * Linear_dim
        x_8 = self.relu(self.Linear(x_7.transpose(1, 2)))
        #Shared_BLSTMP_1 (Batch * speaker) * Time * Linear_dim =〉(Batch * speaker) * Time * (Linear_Shared_layer1_dim * 2)
        lens = [n for n in nframes for i in range(speaker)] 
        x_9 = self.Shared_BLSTMP_1(x_8, lens)
        #Shared_BLSTMP_2 (Batch * speaker) * Time * (Linear_Shared_layer1_dim * 2) =〉 (Batch * speaker) * Time * (Linear_Shared_layer2_dim * 2)
        x_10 = self.Shared_BLSTMP_2(x_9, lens)

        #Combine-Speaker: (Batch * speaker) * Time * (Linear_Shared_layer2_dim * 2) => Batch * Time * (speaker * Linear_Shared_layer2_dim * 2)
        x_11 = x_10.reshape(batchsize, speaker, Time, -1).transpose(1, 2).reshape(batchsize, Time, -1)
        '''
        batchsize * Time * (1stspeaker 2ndspeaker 3rdspeaker 4thspeaker)
        '''

        #BLSTM: Batch * Time * (speaker * Linear_Shared_layer2_dim * 2) => Batch * Time * (BLSTM_Projection_dim * 2)
        x_12 = self.BLSTMP(x_11, nframes)

        #Dimension reduction: remove the padding frames; Batch * Time * (BLSTM_Projection_dim * 2) => (Batch * Time) * (BLSTM_Projection_dim * 2)
        lens = [k for i, m in enumerate(nframes) for k in range(i * Time, m + i * Time)]
        x_13 = x_12.reshape(batchsize * Time, -1)[lens, :]
        '''
        1stBatch(1st sentence) length1 * (BLSTM_size * 2)
        2ndBatch(2nd sentence) length2 * (BLSTM_size * 2)
        ...
        '''
        '''
        for i in self.FC:
            out.append(self.FC[i](x_13))
        '''
        out = []
        out.append(self.FC(x_13))
        return out

