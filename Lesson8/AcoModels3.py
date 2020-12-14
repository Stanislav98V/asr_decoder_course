#!/usr/bin/python
import sys
import os
import numpy as np
import pickle
import scipy
from   scipy import cluster
import gmm
import copy


# python3 trainAM2.py dan out_model2 1
# python3 Task1.py

class AcoModel:
    def dist(self, ftrInput):
    #вычисление дистанции
        return -4.0 * self.gmm.loglikelyhood(ftrInput)

    #training
    def __init__(self, fnSortFile, num_of_mixes): #to train model
        self.name = os.path.basename(fnSortFile).replace('.npy','') #remove extension # Возвращает базовое имя пути
        self.num_of_average = (scipy.cluster.vq.kmeans((np.load(fnSortFile)), num_of_mixes))[0]
        vectors = np.load(fnSortFile)
        clusters = find_cluster(vectors, num_of_mixes)
        means = []
        vars = []
        weights = []
        for key in clusters:
            means.append(np.mean(np.array(clusters[key]),axis=0))
            vars.append(np.var(np.array(clusters[key]),axis=0))
            weights.append(len(clusters[key])/len(vectors))
        means = np.array(means)
        vars = np.array(vars)
        weights = np.array(weights)
        self.gmm = gmm.DiagGauss(weights=weights, means=means, vars=vars)



class AcoModelSet:
    def findModel(self, modelName):
        return self.name2model[modelName]

    def save(self, fname):
        with open(fname, "wb") as f:
            pickle.dump(self, f)

    def __init__(self, sortDir, saveToFile, num_of_mixes):
        self.name2model={}
        print("Training...")
        for fnModel in os.listdir(sortDir): # Возвращает название файлов директории
            model = AcoModel(os.path.join(sortDir, fnModel), num_of_mixes) # Соеденяем имя файла и путь, вызываем AcoModel
            self.name2model[model.name] = model # присваиваем модели имя


        print("...saving")
        self.save(saveToFile)
        print("Done")

    def show(self):
        for name, model in self.name2model.items(): # Выводит имя и среднее значение
            print("Name: {}; gmm: {}".format(name, model.gmm))


def loadAcoModelSet(fname): # Просмотреть содержимое файла
    with open(fname,"rb") as f:
        q = pickle.load(f)
        return q

def find_cluster(vectors, num_of_mixes):
    centers_clusters = (scipy.cluster.vq.kmeans(vectors, num_of_mixes))[0]
    clusters = {}
    for i in range(num_of_mixes):
        clusters[i] = []
    for vector in vectors:
        dist = []
        for center in centers_clusters:
            dist += [np.linalg.norm(vector - center)]
        best_dist = min(dist)
        count = 0
        while count < len(dist):
            if best_dist == dist[count]:
                clusters[count].append(vector)
                count += 1
            else:
                count += 1
    return clusters