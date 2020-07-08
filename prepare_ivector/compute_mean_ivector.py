# -*- coding: utf-8 -*-

import sys

import numpy as np

import HTK


def load_fea(self, path, start, end):
    nSamples, sampPeriod, sampSize, parmKind, data = HTK.readHtk_start_end(path, start, end)
    htkdata = np.array(data).reshape(end - start, int(sampSize / 4))
    return end - start, htkdata


lists = sys.argv[1]
output = sys.argv[2]
ivector = []
IO = open(lists)
for l in IO:
    l = l.rstrip()
    ivector.append(load_
    fea(l).reshape(-1))
    np.save(output, np.mean(np.vstack(ivector), 0))
    IO.close()
