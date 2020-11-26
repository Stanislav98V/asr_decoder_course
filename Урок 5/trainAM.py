# -*- coding: utf-8 -*-

from sys import argv
import AcoModels

a ,in_sort_dir, out_AM_file = argv
# in_sort_dir = 'finaz'
# out_AM_file = 'eeeeeeeer.txt'
e = AcoModels.AcoModelSet(in_sort_dir, out_AM_file)

e.show()