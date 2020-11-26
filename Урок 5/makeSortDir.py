# -*- coding: utf-8 -*-
import numpy as np
import os
from sys import argv

a, ali_file_name, ftr_ark_file, out_sort_dir = argv

ali = []
with open(f'{ali_file_name}') as file: # Открываем файл разметки
    for line in file: # Проходим по строкам
        ali += [line.split()] # Записываем в массив, каждая ечейка массива хранит название записи с ячейками значений

mfcc_signs = []
line_mfcc_signs = []
if not ftr_ark_file.startswith('ark,t:'): # Проверяем префикс
    raise ValueError("only 'ark,t' type is supported")
else:
    with open(f'{ftr_ark_file}') as file: # Открываем файл с коэффициентами
        for line in file: # Проходим по строкам
            if len(line) > 2 and line[-2] == '[': # Проверяем на название записи
                line_mfcc_signs += line[:-2].split() # Добавляем название в массив
            elif line[-1] != ']' and line[-2] != ']': # Проверяем не последний ли это элемент записи "]"
                line_mfcc_signs += [line[:-1].split()] # Добавляем элемент в массив
            else: # Добавляем запись в конечный массив записей
                mfcc_signs += [line_mfcc_signs]
                line_mfcc_signs = [] # Очищаем массив для записи

sort_files = {} # Создаём словарь для сортированных морфем
for record in mfcc_signs: # Проходим по признакам каждой записи
    for ali_record in ali: # Проходим по разметкам каждой записи
        if record[0] == ali_record[0]: # Если записи одинаковые
            count = 1
            while count < len(ali_record): # Создаём цикл количество проходов которого, равно количеству элементов записи без названия
                if ali_record[count] in sort_files: # Если в словаре есть ключ данная морфема
                    sort_files[ali_record[count]] += [record[count]] # Добавляем под ключ значение
                else: # Если такого ключа нет
                    sort_files[ali_record[count]] = [] # Создаём ключ
                    sort_files[ali_record[count]] += [record[count]] # Добавляем под ключ значение
                count += 1

os.mkdir(f'{out_sort_dir}') # Создаём дирректорию под сортировку морфем
for key in sort_files: # Проходим по морфемам
    direct = f'{out_sort_dir}/{key}.txt' # Создаём имя файла для морфемы
    file = open(f'{direct[0]}', 'w') # Создаём файл для морфемы и открываем его
    file.write(f'{sort_files[key]}') # Записываем в файл значения