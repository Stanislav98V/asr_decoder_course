import AcoModels3
import numpy as np
import time
import FtrFile
import GaussMixCompCache
import wfst
start_time = time.time()

class Token:
    def __init__(self, state, dist=0.0, sentence=""):
        self.state = state
        self.dist = dist
        self.sentence = sentence
        self.is_alive = None

def print_token(token):
    print("Token on state #{} dist={} sentence={}".format(token.state,
                                                          token.dist,
                                                          token.sentence))

def print_tokens(tokens):
    print("*** DEBUG. TOKENS LIST ***")
    for token in tokens:
        print_token(token)
    print("*** END DEBUG. TOKENS LIST ***")

def beam_prunning(tokens, thr_common):
    best_token = tokens[np.argmin([i.dist for i in tokens if i.is_alive != False])]
    for token in tokens:
        if token.is_alive != False:
            if best_token.dist + thr_common < token.dist:
                token.is_alive = False
    return tokens

def state_prunning(tokens, best_tokens):
    for token in tokens:
        if best_tokens[token.state] == None \
                or best_tokens[token.state].dist > token.dist:
            if best_tokens[token.state] != None:
                best_tokens[token.state].is_alive = False
            best_tokens[token.state] = token
        else:
            token.is_alive = False
    for state in range(fst_graph.NumStates()):
        best_tokens[state] = None
    return tokens

def recognize(filename, features, fst_graph, thr_common, lm_scale, wd_add):
    print("Recognizing file '{}', samples={}".format(filename,
                                                     features.nSamples))
    start_state = fst_graph.Start() # Находим стартовый токен
    print('Start state = ', start_state)
    active_tokens = [Token(start_state), ] # Создаём токен
    rt = GaussMixCompCache.GaussMixCompCache(AcoModels3.AcoModelSet, AcoModels3.AcoModel)
    next_tokens = []
    best_tokens = {} # Словарь для state_prunning, ключ это номер стейта, значение это лучший токен данного стейта
    for state in range(fst_graph.NumStates()):
        best_tokens[state] = None
    for frame in range(features.nSamples): # Перебираем все векторы записи
        current_frame_ftr = features.readvec()
        GaussMixCompCache.GaussMixCompCache.setFrame(rt, ftr=current_frame_ftr)
        for token in active_tokens: # Перебираем активные токены
            if token.is_alive != False:
                id1,id2 = fst_graph.GetArcIdRange(token.state)
                for arc_id in range(id1,id2): # Проходим все дуги стейта
                    arc = fst_graph.GetArcById(arc_id)
                    if wfst.invertSymbolTable(fst_graph.InputSymbols)[arc.ilabel] != '<eps>':
                        new_token = Token(arc.nextstate)
                        new_token.dist += token.dist + arc.weight * lm_scale # Прибавляем к дистанции предидущую дистанцию и вес дуги умноженный на коэф
                        new_token.dist += GaussMixCompCache.GaussMixCompCache.getDist(rt, arc.ilabel)
                        new_token.sentence += token.sentence # Прибавляем накопленные слова
                        if type(fst_graph.Final(token.state)) == int or float: # Если стейт имеет вес, прибавляем его
                            new_token.dist += fst_graph.Final(arc.nextstate) * lm_scale
                        if wfst.invertSymbolTable(fst_graph.OutputSymbols)[arc.olabel] != '<eps>': # Если на выходе дуги слово
                            if token.state != new_token.state: # И предидущий стейт не равен текущему
                                new_token.dist += wd_add # Прибавляем штраф за слово
                                new_token.sentence += wfst.invertSymbolTable(fst_graph.OutputSymbols)[arc.olabel] + ' ' # Прибавляем слово
                    else: # Если <eps> на входе дуги, проходим насквозь
                        new_token = Token(arc.nextstate)
                        new_token.dist += token.dist
                        new_token.sentence += token.sentence
                    next_tokens.append(new_token)
        next_tokens = state_prunning(next_tokens, best_tokens)
        if len(next_tokens) > 0:
            next_tokens = beam_prunning(next_tokens, thr_common)
        active_tokens = next_tokens
        next_tokens = []
    print_tokens(active_tokens)
    if len(active_tokens) != 0:
        best_token_ol = active_tokens[np.argmin([i.dist for i in active_tokens])]
        print("FINAL!!!!  ", best_token_ol.dist, best_token_ol.sentence)
    else:
        print('No final token')
if __name__ == "__main__":
    InputSymbols = wfst.readSymbolTable("phones.txt")
    OutputSymbols = wfst.readSymbolTable("phones.txt")
    fst_graph = wfst.Wfst("daNetNoCyclic_4.fst", InputSymbols, OutputSymbols)
    records = "ark,t:testNet.txtftr" # Считываем данные из файла запись
    thr_common = 300
    wd_add = 200
    lm_scale = 100
    for filename, features in FtrFile.FtrDirectoryReader(records):
        recognize(filename, features, fst_graph, thr_common, lm_scale, wd_add) # Берём читалку и имя файла из записи и засовываем в recognize
    print("--- %s seconds ---" % (time.time() - start_time))