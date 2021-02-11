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

def recognize(filename, features, fst_graph, thr_common, lm_scale):
    print("Recognizing file '{}', samples={}".format(filename,
                                                     features.nSamples))
    start_state = fst_graph.Start()
    active_tokens = [Token(start_state), ] # Создаём токен
    next_tokens = []
    best_tokens = {}
    for state in range(fst_graph.NumStates()):
        best_tokens[state] = None
    for frame in range(features.nSamples): # Перебираем все векторы записи
        for token in active_tokens: # Перебираем активные токены
            if token.is_alive != False:
                id1,id2 = fst_graph.GetArcIdRange(token.state)
                for arc_id in range(id1,id2):
                    arc = fst_graph.GetArcById(arc_id)
                    new_token = Token(arc.nextstate)
                    new_token.dist +=  token.dist + arc.weight * lm_scale
                    # if fst_graph.Final(arc.nextstate) != 'inf':
                    #     new_token.dist += fst_graph.Final(arc.nextstate)
                    new_token.sentence += token.sentence + wfst.invertSymbolTable(fst_graph.OutputSymbols)[arc.olabel]
                    next_tokens.append(new_token)
        next_tokens = state_prunning(next_tokens, best_tokens)
        next_tokens = beam_prunning(next_tokens, thr_common)
        active_tokens = next_tokens
        next_tokens = []
    print_tokens(active_tokens)
    best_token_ol = active_tokens[np.argmin([i.dist for i in active_tokens])]
    print("FINAAAAAAAAAL!!!!  ", best_token_ol.dist, best_token_ol.sentence)

if __name__ == "__main__":
    InputSymbols = wfst.readSymbolTable("isyms.txt")
    OutputSymbols = wfst.readSymbolTable("osyms.txt")
    fst_graph = wfst.Wfst("danet.fst", InputSymbols,OutputSymbols)
    records = "ark,t:test_MFCC.txtftr" # Считываем данные из файла запись
    thr_common = 200
    lm_scale = 0.9
    for filename, features in FtrFile.FtrDirectoryReader(records):
        recognize(filename, features, fst_graph, thr_common, lm_scale) # Берём читалку и имя файла из записи и засовываем в recognize
    print("--- %s seconds ---" % (time.time() - start_time))