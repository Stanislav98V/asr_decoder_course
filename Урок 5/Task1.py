#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import AcoModels
import numpy as np
import cProfile
import time
import FtrFile
import FtrFile2
start_time = time.time()

#########################################
# State, Graph, etc...
#########################################

class State:
    def __init__(self, model, idx): # idx is for debug purposes
        self.model = model
        self.word = None
        self.isFinal = False
        self.nextStates = []
        self.idx = idx
        self.nextStatesIdxs = []
        self.best_token = None



def print_state(state):
    nextStatesIdxs = [s.idx for s in state.nextStates]
    state.nextStatesIdxs=nextStatesIdxs
    print("State: idx={} word={} isFinal={} nextStatesIdxs={} model={}".format(
        state.idx, state.word, state.isFinal, state.nextStatesIdxs, state.model))


def load_graph(rxfilename):
    startState = State(None, 0) # Создаём первый элемент графа
    graph = [startState, ] # Записываем его в Спиок узлов графа
    stateIdx = 1
    for word, features in FtrFile2.FtrDirectoryReader(rxfilename): # Получили имя и значения
        prevState = startState
        for frame in range(features.nSamples):
            state = State(features.readvec(), stateIdx) # Создаём стейт с индексом и значением
            state.nextStates.append(state) # add loop # Добавляем переход в себя
            prevState.nextStates.append(state) # Добавляем переход в иекущий стейт для prevState он равен прошлому стейту
            prevState = state # prevState тепер ссылается на текущий стейт
            print_state(state) # Добавляем стейту все нужные значения и выводим его
            graph.append(state)
            # if frame >= 2 :
            #     graph[stateIdx - 2].nextStates.append(state)
            # if frame >= 3:
            #     graph[stateIdx - 3].nextStates.append(state)
            stateIdx += 1
        if state:
            state.word = word
            state.isFinal = True
        print_state(graph[0])


    return graph # Создаём граф


def check_graph(graph):
    assert len(graph) > 0, "graph is empty."
    assert graph[0].model is None \
        and graph[0].word is None \
        and not graph[0].isFinal, "broken start state in graph."
    idx = 0
    for state in graph:
        assert state.idx == idx
        idx += 1
        assert (state.isFinal and state.word is not None) \
            or (not state.isFinal and state.word is None)


def print_graph(graph):
    print("*** DEBUG. GRAPH ***")
    np.set_printoptions(formatter={'float': '{: 0.1f}'.format})
    for state in graph:
        print_state(state)
    print("*** END DEBUG. GRAPH ***")


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




def recognize(filename, features, graph, thr_common):
    print("Recognizing file '{}', samples={}".format(filename,
                                                     features.nSamples))

# мой код
    start_state = graph[0]
    active_tokens = [Token(start_state), ] # Создаём токен
    next_tokens = []
    for frame in range(features.nSamples): # Перебираем все векторы записи
        current_frame_ftr = features.readvec()
        for token in active_tokens: # Перебираем активные токены
            if token.is_alive != False:
                for next_state_id in token.state.nextStatesIdxs: # Перебираем возможные пути. Я чутка изменил код, сделал naxeStatesIdxs  атрибутом класса State
                    new_token = Token(graph[next_state_id]) # Создаём новый токен в указанном узле
                    new_token.dist += token.dist # Копируем расстояние в новый токен
                    lv = AcoModels.AcoModel.dist(graph[next_state_id].model, current_frame_ftr) # Считаем расстояния
                    new_token.dist += lv
                    next_tokens.append(new_token)
        next_tokens = state_prunning(next_tokens)
        next_tokens = beam_prunning(next_tokens, thr_common)
        active_tokens = next_tokens
        next_tokens = []
    final_best_tokens=[]
    # MAGIC
    # не мой код
    print_tokens(active_tokens) # Выводим активынй токен
    # мой код

    for token in active_tokens:
        if token.is_alive != False and token.state.isFinal == True:
            final_best_tokens += [token]
    if len(final_best_tokens) > 0:
        best_token_ol = final_best_tokens[np.argmin([i.dist for i in final_best_tokens])]
        str_out = f"Minimum distance={best_token_ol.dist} with isFinal={best_token_ol.state.isFinal} " \
                f"and is_alive={best_token_ol.is_alive} and word={best_token_ol.state.word}"
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
    records = "ark,t:file_tain.txtftr" # Считываем данные из файла запись
    with open('OTV.txt', 'w') as f: # записываем ответ в файл
        f.write('')
    graph = load_graph(etalons) # Создаём граф
    check_graph(graph) # Провека графа
    print_graph(graph) # Вывод графа

    for filename, features in FtrFile.FtrDirectoryReader(records):
        recognize(filename, features, graph, thr_common) # Берём читалку и имя файла из записи и засовываем в recognize
        #cProfile.run('recognize(filename, features, graph)')

    print("--- %s seconds ---" % (time.time() - start_time))
