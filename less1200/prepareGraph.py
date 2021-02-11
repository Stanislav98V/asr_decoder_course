from sys import argv
import wfst
import os

# python3 prepareGraph.py LG.fst

a, file_fst = argv
os.system(f'/home/stas/kaldi/src/fstbin/fstrmsymbols phones.disamb.codes {file_fst} Output.fst')
os.system('/home/stas/disamb2 --add-loop --no-aux-processing Output.fst Output.fst')
os.system('fstprint Output.fst  Output_fin.fst')
