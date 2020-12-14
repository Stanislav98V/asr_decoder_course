# -*- coding: utf-8 -*-

from sys import argv
import AcoModels3
import gmm

a, in_sort_dir, out_AM_file, num_of_clusters = argv
# in_sort_dir = 'finaz'
# out_AM_file = 'eeeeeeeer.txt'
e = AcoModels3.AcoModelSet(in_sort_dir, out_AM_file, int(num_of_clusters))

# e.show()

r = AcoModels3.loadAcoModelSet(f'{out_AM_file}')
for name, model in r.name2model.items():  # Выводит имя и среднее значение
    print("Name: {}; Average: {}".format(name, model.num_of_average))