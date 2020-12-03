#!/usr/bin/python
import sys
import os
import numpy as np
import pickle
import scipy
from scipy import cluster


class AcoModel:
    def dist(self, ftrInput):
    #вычисление дистанции
        distiki = []
        for i in self.num_of_average:
            distiki += [np.linalg.norm(i - ftrInput)]
        return np.min(distiki)

    #training
    def __init__(self, fnSortFile, num_of_clusters): #to train model
        self.name = os.path.basename(fnSortFile).replace('.npy','') #remove extension # Возвращает базовое имя пути
        self.num_of_average = (scipy.cluster.vq.kmeans((np.load(fnSortFile)), num_of_clusters))[0]
        self.distortion = (scipy.cluster.vq.kmeans((np.load(fnSortFile)), num_of_clusters))[1]



class AcoModelSet:
    def findModel(self, modelName):
        return self.name2model[modelName]

    def save(self, fname):
        with open(fname, "wb") as f:
            pickle.dump(self, f)

    def __init__(self, sortDir, saveToFile, num_of_clusters):
        self.name2model={}
        print("Training...")
        for fnModel in os.listdir(sortDir): # Возвращает название файлов директории
            model = AcoModel(os.path.join(sortDir, fnModel), num_of_clusters) # Соеденяем имя файла и путь, вызываем AcoModel
            self.name2model[model.name] = model # присваиваем модели имя


        print("...saving")
        self.save(saveToFile)
        print("Done")

    def show(self):
        for name, model in self.name2model.items(): # Выводит имя и среднее значение
            print("Name: {}; Averages: {}: Distortion: {}".format(name, model.num_of_average, model.distortion))


def loadAcoModelSet(fname): # Просмотреть содержимое файла
    with open(fname,"rb") as f:
        q = pickle.load(f)
        return q
