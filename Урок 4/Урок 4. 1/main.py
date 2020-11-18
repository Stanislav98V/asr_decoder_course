#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import numpy as np

import FtrFile


#########################################
# State, Graph, etc...
#########################################

class State:
    best_token=None
    def __init__(self, ftr, idx): # idx is for debug purposes
        self.ftr = ftr
        self.word = None
        self.isFinal = False
        self.nextStates = []
        self.idx = idx
        self.nextStatesIdxs = []



def print_state(state):
    nextStatesIdxs = [s.idx for s in state.nextStates]
    # if len(nextStatesIdxs) > 1 and :
    #     nextStatesIdxs += [state.idx + 2]
    state.nextStatesIdxs=nextStatesIdxs
    print("State: idx={} word={} isFinal={} nextStatesIdxs={} ftr={} ".format(
        state.idx, state.word, state.isFinal, state.nextStatesIdxs, state.ftr))


def load_graph(rxfilename):
    startState = State(None, 0) # Создаём первый элемент графа
    graph = [startState, ] # Записываем его в Спиок узлов графа
    stateIdx = 1
    for word, features in FtrFile.FtrDirectoryReader(rxfilename): # Получили имя и значения
        prevState = startState
        for frame in range(features.nSamples):
            state = State(features.readvec(), stateIdx)
            state.nextStates.append(state) # add loop
            prevState.nextStates.append(state)
            prevState = state
            print_state(state)
            graph.append(state)
            stateIdx += 1
        if state:
            state.word = word
            state.isFinal = True
        print_state(graph[0])

    # Начало кода к 4 уроку

    len_graph = len(graph)
    counter = 0
    graph_word_idx = []
    for token in graph:
        if token.word != None:
            graph_word_idx += [token]
    for token_i in graph:
        counter += 1
        counter_i = 0
        counter_i_i = 0
        if token_i.idx != 0 and token_i.word == None:
            for token_i_i in graph_word_idx:
                if token_i_i.idx == token_i.idx + 1:
                    counter_i += 1
                    counter_i_i +=1
                elif token_i_i.idx == token_i.idx + 2:
                    counter_i_i += 1
            if len_graph - token_i.idx > 1 and counter_i == 0:
                for token_i_i in graph:
                    if token_i_i.idx == counter + 1:
                        token_i.nextStates.append(token_i_i)
            if len_graph - token_i.idx > 1 and counter_i_i == 0:
                for token_i_i in graph:
                    if token_i_i.idx == counter + 2:
                        token_i.nextStates.append(token_i_i)
        counter_i = 0
        counter_i_i = 0

    # Конец кода к 4 уроку

    return graph # Создаём граф


def check_graph(graph):
    assert len(graph) > 0, "graph is empty."
    assert graph[0].ftr is None \
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
    distan = 0
    le = len(ftr)
    for i in range(le):
        distan += (current_frame_ftr[i] - ftr[i]) ** 2
    distan = distan ** 0.5
    return distan # Считаем расстояние евклидовой метрикой


# не мой код

def state_prune(tokes):
    length_graph = len(graph)
    for graph_idx in range(length_graph):
        tokens_graph_idx = [i for i in tokes if i.state.idx == graph_idx]
        if len(tokens_graph_idx) > 0:
            best_token_graph_idx = tokens_graph_idx[np.argmin([i.dist for i in tokens_graph_idx if i.is_alive != False])]
            State.best_token = best_token_graph_idx
            for token in tokens_graph_idx:
                if token == best_token_graph_idx:
                    token.is_alive = True
                else:
                    token.is_alive = False
    return tokes





def recognize(filename, features, graph):
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
                    new_token.dist += compute_distance(current_frame_ftr, graph[next_state_id].ftr) # Считаем расстояния
                    next_tokens.append(new_token)
        next_tokens = state_prune(next_tokens)
        active_tokens = next_tokens
        next_tokens = []

    final_best_tokens=[]
    # MAGIC
    # не мой код
    print_tokens(active_tokens) # Выводим активынй токен
    # мой код
    for token in active_tokens:
        if token.is_alive == True and token.state.isFinal == True:
            final_best_tokens += [token]
    State.best_token = final_best_tokens[np.argmin([i.dist for i in final_best_tokens])]
    str_out = f"Minimum distance={State.best_token.dist} with isFinal={State.best_token.state.isFinal} " \
              f"and is_alive={State.best_token.is_alive} and word={State.best_token.state.word}"
    print(str_out) # выводим ответ с минимальным расстоянием и названием эталона
    with open('OTV.txt', 'a') as f: # записываем ответ в файл
        f.write(str_out)


#########################################
# Main
#########################################

if __name__ == "__main__":
    etalons = "ark,t:etalons_mfcc.txtftr" # Считываем данные из файла эталона
    records = "ark,t:record_mfcc.txtftr" # Считываем данные из файла запись
    with open('OTV.txt', 'w') as f: # записываем ответ в файл
        f.write('')
    graph = load_graph(etalons) # Создаём граф
    check_graph(graph) # Провека графа
    print_graph(graph) # Вывод графа

    for filename, features in FtrFile.FtrDirectoryReader(records):
        recognize(filename, features, graph) # Берём читалку и имя файла из записи и засовываем в recognize
