from sys import argv
import wfst


if len(argv) > 2:
    a, file_fst, isyms_txt, osyms_txt = argv
else:
    a, file_fst = argv
    isyms_txt = 'phones.txt'
    osyms_txt = 'phones.txt'


InputSymbols = wfst.readSymbolTable(isyms_txt)
OutputSymbols = wfst.readSymbolTable(osyms_txt)
fst = wfst.Wfst(file_fst, InputSymbols, OutputSymbols)


for id_state in range(fst.NumStates()):
    id1, id2 = fst.GetArcIdRange(id_state)
    for arc_id in range(id1, id2):
           arc = fst.GetArcById(arc_id)
           if len(argv) > 2:
                print(id_state, arc.nextstate, wfst.invertSymbolTable(fst.InputSymbols)[arc.ilabel],
                      wfst.invertSymbolTable(fst.OutputSymbols)[arc.olabel], arc.weight, sep='\t')
           else:
                print(id_state, arc.nextstate, arc.ilabel,
                      arc.olabel, arc.weight, sep='\t')