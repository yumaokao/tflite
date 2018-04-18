#!/usr/bin/env python

import numpy as np


def splitter(name):
    inpy = np.load('{}.npy'.format(name))
    print(inpy.shape)

    for b in range(inpy.shape[0]):
        bfn = '{}_{}.npy'.format(name, b)
        nnpy = inpy[b][:][:][:]
        nnpy = np.expand_dims(nnpy, axis=0)
        # print(nnpy.shape)
        print(bfn)
        np.save(bfn, nnpy)


splitter('./batch_xs')
splitter('./output_ys')
