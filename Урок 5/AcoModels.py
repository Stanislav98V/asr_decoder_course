#!/usr/bin/python
import sys
import os
import numpy as np
import pickle


class AcoModel:
    def dist(self, ftrInput):
    #вычисление дистанции
        return np.linalg.norm(self.average - ftrInput) # Вычисление дистанции

    #training
    def __init__(self, fnSortFile): #to train model
        self.name = os.path.basename(fnSortFile).replace('.npy','') #remove extension # Возвращает базовое имя пути
        self.average = np.average(np.load(fnSortFile),axis=0) # Возвращаем массив из файлов # Вычисляет среднее арефметическое



class AcoModelSet:
    def findModel(self, modelName):
        return self.name2model[modelName]

    def save(self, fname):
        with open(fname, "wb") as f:
            pickle.dump(self, f)

    def __init__(self, sortDir, saveToFile):
        self.name2model={}
        print("Training...")
        for fnModel in os.listdir(sortDir): # Возвращает название файлов директории
            model = AcoModel(os.path.join(sortDir, fnModel)) # Соеденяем имя файла и путь, вызываем AcoModel
            self.name2model[model.name] = model # присваиваем модели имя


        print("...saving")
        self.save(saveToFile)
        print("Done")

    def show(self):
        for name, model in self.name2model.items(): # Выводит имя и среднее значение
            print("Name: {}; Average: {}".format(name, model.average))


def loadAcoModelSet(fname): # Просмотреть содержимое файла
    with open(fname,"rb") as f:
        q = pickle.load(f)
        return q
