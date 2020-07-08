# -*- coding: utf-8 -*-

import torch


def SoftCrossEntropy(inputs, target, reduction='mean'):
    '''
    inputs: Time * Num_class 
    target: Time * Num_class
    '''
    log_likelihood = -torch.nn.functional.log_softmax(inputs, dim=-1)
    batch_size = inputs.shape[0]
    loss = torch.sum(torch.mul(log_likelihood, target))
    if reduction == 'mean':
        loss /= batch_size
    return loss


def SoftCrossEntropy_4Targets(ypreds, label_data):
    loss = sum(map(lambda n: SoftCrossEntropy(ypreds[n], label_data[n]), range(4)))
    return loss


def CrossEntropy_SingleTargets(y_preds, label):
    criterion = torch.nn.CrossEntropyLoss()
    loss = sum(map(lambda n: criterion(y_preds[n], label[n]), range(len(y_preds))))
    return loss
