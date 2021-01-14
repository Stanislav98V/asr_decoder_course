import numpy as np
import FtrFile2
import AcoModels3

class State:
    def __init__(self, model, idx, word=None): # idx is for debug purposes
        self.model = model
        self.word = None
        self.isFinal = False
        self.nextStates = []
        self.idx = idx
        self.nextStatesIdxs = []
        self.best_token = None
        self.ugid = None

    def isAlmostEqual(self, q):
        return (self.model == q.model) and (self.isFinal == q.isFinal) and (self.word == q.word)

def print_state(state):
    nextStatesIdxs = [s.idx for s in state.nextStates]
    state.nextStatesIdxs=nextStatesIdxs
    print("State: idx={} word={} isFinal={} nextStatesIdxs={}".format(
        state.idx, state.word, state.isFinal, state.nextStatesIdxs))


def load_graph(rxfilename):
    mod = AcoModels3.loadAcoModelSet('out_model2')
    startState = State(None, 0) # Создаём первый элемент графа
    graph = [startState, ] # Записываем его в Спиок узлов графа
    stateIdx = 1
    for word, features in FtrFile2.FtrDirectoryReader(rxfilename): # Получили имя и значения
        prevState = startState
        #transcription = ''
        for frame in range(features.nSamples):
            state = State(features.readvec(), stateIdx) # Создаём стейт с индексом и значением
            # transcription += state.model.name + " "
            # if prevState != startState:
            #     addWord(graph, prevState, state, state.word, transcription, mod)
            state.nextStates.append(state) # add loop # Добавляем переход в себя
            prevState.nextStates.append(state) # Добавляем переход в текущий стейт для prevState он равен прошлому стейту


            #transcription += state.model.name + " "
            # if prevState != startState:
            #      addWord(graph, prevState, state, word, transcription, mod)
            # ВОЗМОЖНО НУЖНО ВСТАВИТЬ СЮДА КОД

            prevState = state # prevState тепер ссылается на текущий стейт
            print_state(state) # Добавляем стейту все нужные значения и выводим его
            graph.append(state)
            stateIdx += 1
        if state:
            state.word = word
            state.isFinal = True
            state.nextStates.append(startState)
        print_state(graph[0])
    return graph # Создаём граф

def addSubState(graph, state_perent, new_state):
   for stChild in state_perent.nextStates:
      if stChild.isAlmostEqual(new_state):
          return stChild
   graph.append(new_state)
   state_perent.nextStates.append(new_state)
   return new_state
pass

def addWord(graph, before_start_state, after_end_state, word, transcription, mod): # Добавляет слово в граф
   #mod = AcoModels3.loadAcoModelSet('out_model2')
   prevState = before_start_state # Перед началом State
   modelNames = transcription.split() # Соеденяем транскрипцию
   idx = after_end_state.idx
   for modelName, iModel in zip(modelNames, range(len(modelNames))):
       wordId = None
       ugid = opr_fon(modelName, mod)
       model = opr_fon2(ugid, mod)
       if iModel == len(modelNames) - 1:
           wordId = word
       rty = opr_fon(modelName, mod)
       print(modelNames,len(modelNames), idx,  'PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP')
       state = State(opr_fon2(rty, mod), idx + iModel - len(modelNames) + 1, wordId)
       state.nextStates.append(state) #self-loop
       prevState = addSubState(graph, prevState, state)
   pass
   prevState.nextStates.append(after_end_state)
pass


def opr_fon(name_mod, mod):
    count = 0
    for key, model in mod.name2model.items():
        if name_mod == key:
            return count
        count += 1
    return count

def opr_fon2(ugid, mod):
    count = 0
    for key, model in mod.name2model.items():
        if count == ugid:
            return model
        count += 1
    return model

def check_graph(graph):
    assert len(graph) > 0, "graph is empty."
    assert graph[0].model is None \
        and graph[0].word is None \
        and not graph[0].isFinal, "broken start state in graph."
    idx = 0
    # for i in graph:
    #     print_state(i)
    for state in graph:
        # assert state.idx == idx
        idx += 1
        assert (state.isFinal and state.word is not None) \
            or (not state.isFinal and state.word is None)


def print_graph(graph):
    print("*** DEBUG. GRAPH ***")
    np.set_printoptions(formatter={'float': '{: 0.1f}'.format})
    for state in graph:
        print_state(state)
    print("*** END DEBUG. GRAPH ***")


def load_dict(dict):
    graph = load_graph(dict)
    check_graph(graph)  # Провека графа
    print_graph(graph)  # Вывод графа
    return graph




