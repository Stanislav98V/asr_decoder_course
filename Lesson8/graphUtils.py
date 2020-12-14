import numpy as np
import FtrFile2

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
            stateIdx += 1
        if state:
            state.word = word
            state.isFinal = True
            state.nextStates.append(startState)
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


def load_dict(dict):
    graph = load_graph(dict)
    check_graph(graph)  # Провека графа
    print_graph(graph)  # Вывод графа
    return graph




