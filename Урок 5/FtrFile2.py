# -*- coding: utf-8 -*-
import AcoModels
import numpy as np

def FtrDirectoryReader(rxfilename):
    path = rxfilename
    for fname, reader in ArkReader(path):
        yield fname, reader


def ArkReader(path):
    with open(path, 'r') as ark_t_file:
        for line in ark_t_file:
            line = line.strip()
            line = line.split()
            if not line:
                continue
            yield line[0], MatrixReader(line[1:])


class MatrixReader:
    def __init__(self, ark_t_file):
        lines = []
        mod = AcoModels.loadAcoModelSet('eeeeeeeer.txt')
        for line in ark_t_file:
            line = line.strip()
            line = line.split()
            if not line:
                continue
            line = str(line)
            line = line[2:-2]
            line = mod.name2model[line]
            lines.append(line)
        self.models = np.atleast_2d(lines)
        expectNSamples = len(lines)
        self.nDim, self.nSamples = self.models.shape
        if expectNSamples != self.nSamples:
            assert expectNSamples == self.nDim, \
                """ something wrong with features dimension """
            self.models = self.models.transpose()
            self.nDim, self.nSamples = self.models.shape
        self.__returnedVector__ = -1




    def readvec(self):
        self.__returnedVector__ += 1
        return self.models[0][self.__returnedVector__]

    def getall(self):
        return self.models