#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import AcoModels3
import numpy as np
import cProfile
import time
import FtrFile
import FtrFile2
import graphUtils
start_time = time.time()

#########################################
# Token
#########################################

class Token:
    def __init__(self, state, dist=0.0, sentence=""):
        self.state = state
        self.dist = dist
        self.sentence = sentence
        self.is_alive = None


def print_token(token):
    print("Token on state #{} dist={} isFinal={} sentence={}".format(token.state.idx,
                                                          token.dist,
                                                          token.state.isFinal,
                                                          token.sentence))


def print_tokens(tokens):
    print("*** DEBUG. TOKENS LIST ***")
    for token in tokens:
        print_token(token)
    print("*** END DEBUG. TOKENS LIST ***")

#########################################
# Decoder
#########################################

# мой код

def compute_distance(current_frame_ftr, ftr):
    return sum((current_frame_ftr - ftr) **2 ) ** 0.5 # Считаем расстояние евклидовой метрикой

# не мой код

def state_prunning(tokens):
    for token in tokens:
        if token.state.best_token == None \
                or token.state.best_token.dist > token.dist:
            if token.state.best_token != None:
                token.state.best_token.is_alive = False
            token.state.best_token = token
        else:
            token.is_alive = False
    for state in graph:
        state.best_token = None
    return tokens


def beam_prunning(tokens, thr_common):
    best_token = tokens[np.argmin([i.dist for i in tokens if i.is_alive != False])]
    for token in tokens:
        if token.is_alive != False:
            if best_token.dist + thr_common < token.dist:
                token.is_alive = False
    return tokens

def count_token(next_state_id, token, current_frame_ftr, wd_add):
    new_token = Token(graph[next_state_id])  # Создаём новый токен в указанном узле
    new_token.dist += token.dist  # Копируем расстояние в новый токен
    if token.sentence != new_token.sentence:
        new_token.sentence = token.sentence
    if new_token.state.isFinal == True:
        if token.state != new_token.state:
            new_token.dist += wd_add
            new_token.sentence += new_token.state.word + ' '
    new_token.dist += AcoModels3.AcoModel.dist(graph[next_state_id].model,
                                               current_frame_ftr)  # Считаем расстояния
    return new_token


def recognize(filename, features, graph, thr_common):
    print("Recognizing file '{}', samples={}".format(filename,
                                                     features.nSamples))

# мой код

    start_state = graph[0]
    active_tokens = [Token(start_state), ] # Создаём токен
    next_tokens = []
    wd_add = 120
    for frame in range(features.nSamples): # Перебираем все векторы записи
        current_frame_ftr = features.readvec()
        for token in active_tokens: # Перебираем активные токены
            if token.is_alive != False:
                for next_state_id in token.state.nextStatesIdxs: # Перебираем возможные пути. Я чутка изменил код, сделал naxeStatesIdxs  атрибутом класса State
                    if next_state_id == 0:
                        for next_state_id_two in start_state.nextStatesIdxs:
                            new_token = count_token(next_state_id_two, token, current_frame_ftr, wd_add)
                            next_tokens.append(new_token)
                    else:
                        new_token = count_token(next_state_id, token, current_frame_ftr, wd_add)
                        next_tokens.append(new_token)
        next_tokens = state_prunning(next_tokens)
        #next_tokens = beam_prunning(next_tokens, thr_common)
        active_tokens = next_tokens
        next_tokens = []
    final_best_tokens = []

    # MAGIC
    # не мой код

    print_tokens(active_tokens) # Выводим активынй токен

    # мой код

    for token in active_tokens:
        if token.is_alive != False and token.state.isFinal == True:
            final_best_tokens += [token]
    if len(final_best_tokens) > 0:
        best_token_ol = final_best_tokens[np.argmin([i.dist for i in final_best_tokens])]
        str_out = f"{filename} {best_token_ol.sentence}"
        print(str_out) # выводим ответ с минимальным расстоянием и названием эталона
        with open('OTV.txt', 'a') as f: # записываем ответ в файл
            f.write(str_out + '\n')
    else:
        print('<no-final-token>')
        with open('OTV.txt', 'a') as f: # записываем ответ в файл
            f.write('<no-final-token>\n')

#########################################
# Main
#########################################

if __name__ == "__main__":
    thr_common = 140
    etalons = "file_ali.phoneNames" # Считываем данные из файла эталона
    records = "ark,t:daNetSeqmfcc.txtftr" # Считываем данные из файла запись
    with open('OTV.txt', 'w') as f: # записываем ответ в файл
        f.write('')
    graph = graphUtils.load_dict(etalons)
    for filename, features in FtrFile.FtrDirectoryReader(records):
        recognize(filename, features, graph, thr_common) # Берём читалку и имя файла из записи и засовываем в recognize
    print("--- %s seconds ---" % (time.time() - start_time))