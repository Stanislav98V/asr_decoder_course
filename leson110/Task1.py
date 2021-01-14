#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
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
import GaussMixCompCache
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

def count_token(next_state_id, token, current_frame_ftr, wd_add, rt, mod):
    new_token = Token(graph[next_state_id])  # Создаём новый токен в указанном узле
    new_token.dist += token.dist  # Копируем расстояние в новый токен
    if token.sentence != new_token.sentence:
        new_token.sentence = token.sentence
    if new_token.state.word != None:
        if token.state != new_token.state:
            new_token.dist += wd_add
            new_token.sentence += new_token.state.word + ' '

    # new_token.dist += AcoModels3.AcoModel.dist(graph[next_state_id].model,
    #                                          current_frame_ftr)  # Считаем расстояния
    ugit = opr_fon(graph[next_state_id], mod)# !!!!!!!!!!!!!!!!!!!!
    new_token.dist += GaussMixCompCache.GaussMixCompCache.getDist(rt, ugit)# !!!!!!!!!!!!!!!!!!!!
    return new_token

def  leaveTopNTokens(tokens, N_leavTopTokens):
    count = 0
    if len(tokens) >= N_leavTopTokens - 1:
        tokens.sort(key=lambda x: x.dist)
        for token in tokens:
            if token.is_alive != False:
                count += 1
            if count >= N_leavTopTokens:
                token.is_alive = False
    return tokens

def opr_fon(ugid, mod):
    count = 0
    for key, model in mod.name2model.items():
        if ugid.model.name == key:
            return count
        count += 1
        er = count
    return er

def running_beam(next_tokens, next_state_id, mod, rt, token, thr_common):
    ugit = opr_fon(graph[next_state_id], mod)  # !!!!!!!!!!!!!!!!!!!!
    dist = GaussMixCompCache.GaussMixCompCache.getDist(rt, ugit) + token.dist
    b_d = next_tokens[np.argmin([i.dist for i in next_tokens if i.is_alive != False])]
    #print(b_d.dist, thr_common, dist, 'OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO')
    if b_d.dist + thr_common > dist:
        return 1
    else:
        return 0



def recognize(filename, features, graph, thr_common):
    print("Recognizing file '{}', samples={}".format(filename,
                                                     features.nSamples))
    # мой код

    start_state = graph[0]
    active_tokens = [Token(start_state), ] # Создаём токен
    next_tokens = []
    wd_add = 120
    N_leavTopTokens = 300
    rt = GaussMixCompCache.GaussMixCompCache(AcoModels3.AcoModelSet, AcoModels3.AcoModel)# !!!!!!!!!!!!!!!!!!!!
    mod = AcoModels3.loadAcoModelSet('out_model2')# !!!!!!!!!!!!!!!!!!!!
    z = 0
    for frame in range(features.nSamples): # Перебираем все векторы записи
        current_frame_ftr = features.readvec()
        GaussMixCompCache.GaussMixCompCache.setFrame(rt, ftr=current_frame_ftr) # !!!!!!!!!!!!!!!!!!!!
        count = 0
        for token in active_tokens: # Перебираем активные токены
            if token.is_alive != False:
                for next_state_id in token.state.nextStatesIdxs: # Перебираем возможные пути.
                    if count > 1 and len(next_tokens) > 0:
                        if next_state_id == 0:
                                for next_state_id_two in start_state.nextStatesIdxs:
                                    if running_beam(next_tokens, next_state_id_two, mod, rt, token, thr_common) == 0:
                                        z += 1
                                        count += 1
                                        continue
                                    else:
                                        new_token = count_token(next_state_id_two, token, current_frame_ftr, wd_add, rt,mod)
                                        next_tokens.append(new_token)
                                        count += 1
                        else:
                            if running_beam(next_tokens, next_state_id, mod, rt, token, thr_common) == 0:
                                z += 1
                                count += 1
                                continue
                            else:
                                new_token = count_token(next_state_id, token, current_frame_ftr, wd_add, rt,mod)
                                next_tokens.append(new_token)
                                count += 1
                    else:
                        if next_state_id == 0:
                            for next_state_id_two in start_state.nextStatesIdxs:
                                new_token = count_token(next_state_id_two, token, current_frame_ftr, wd_add, rt, mod)
                                next_tokens.append(new_token)
                                count += 1
                        else:
                            new_token = count_token(next_state_id, token, current_frame_ftr, wd_add, rt, mod)# !!!!!!!!!!!!!!!!!!!!
                            next_tokens.append(new_token)
                            count += 1
        next_tokens = state_prunning(next_tokens)
        next_tokens = beam_prunning(next_tokens, thr_common)
        next_tokens = leaveTopNTokens(next_tokens, N_leavTopTokens)
        active_tokens = next_tokens
        next_tokens = []
    final_best_tokens = []

    # MAGIC
    # не мой код

    print_tokens(active_tokens) # Выводим активынй токен
    print(len(active_tokens))
    # мой код
    print(z, 'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE')
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
    thr_common = 200
    etalons = "OnlyTestWordsNoOOV.dic" # Считываем данные из файла эталона
    records = "ark,t:test10.txtftr" # Считываем данные из файла запись
    with open('OTV.txt', 'w') as f: # записываем ответ в файл
        f.write('')
    graph = graphUtils.load_dict(etalons)
    for filename, features in FtrFile.FtrDirectoryReader(records):
        recognize(filename, features, graph, thr_common) # Берём читалку и имя файла из записи и засовываем в recognize
    print("--- %s seconds ---" % (time.time() - start_time))