#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:38:56 2019
@author: cong
"""

import glob
import os.path
import numpy as np
import os
from itertools import combinations

# 图片数据文件夹
INPUT_DATA = '/run/media/root/0ec8ef97-3812-4bf2-88a2-127cb286cf3e/Desktop/jk/jk/facenet-pytorch-main/datasets/cyclegan_data/fake'


def create_image_lists():
    matched_result = set()
    k = 0
    # 获取当前目录下所有的子目录,这里x 是一个三元组(root,dirs,files)，第一个元素表示INPUT_DATA当前目录，
    # 第二个元素表示当前目录下的所有子目录,第三个元素表示当前目录下的所有的文件
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]

    for sub_dir in sub_dirs[1:]:
        # 获取当前目录下所有的有效图片文件
        extensions = 'bmp'
        # 把图片存放在file_list列表里
        file_list = []
        # os.path.basename(sub_dir)返回sub_sir最后的文件名
        file0_list = []
        dir_name = os.path.basename(sub_dir)
        file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extensions)
        # glob.glob(file_glob)获取指定目录下的所有图片，存放在file_list中
        file_list.extend(glob.glob(file_glob))
        file0_list.extend(glob.glob(file_glob))
        file0_list.sort()

        # 通过目录名获取类别的名称
        label_name = dir_name
        # print(label_name)
        length = len(file_list)
        d_com = []
        d = np.arange(length)
        for i, j in combinations(d, 2):
            d_com.append([i, j])
            base_name1 = file0_list.index(file_list[i]) + 1  # 获取文件的名称
            base_name2 = file0_list.index(file_list[j]) + 1
            s = label_name + '\t' + str(base_name1) + '\t' + str(base_name2)
            if s not in matched_result:
                # 将当前类别的数据放入结果字典
                matched_result.add(label_name + '\t' + str(base_name1) + '\t' + str(base_name2))

        k = k + 1
        # print(len(matched_result))

    # 返回整理好的所有数据
    return matched_result, k


def create_pairs():
    unmatched_result = set()  # 不同类的匹配对
    k = 0
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    # sub_dirs[0]表示当前文件夹本身的地址，不予考虑，只考虑他的子目录
    for sub_dir in sub_dirs[1:]:
        # 获取当前目录下所有的有效图片文件
        extensions = ['png', 'bmp']
        file_list = []
        # 把图片存放在file_list列表里

        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            # glob.glob(file_glob)获取指定目录下的所有图片，存放在file_list中
            file_list.extend(glob.glob(file_glob))

    length_of_dir = len(sub_dirs)

    a = np.arange(1, length_of_dir)
    a_com = []
    for i, j in combinations(a, 2):
        # print(i,j)
        a_com.append([i, j])

        class1 = sub_dirs[i]
        class2 = sub_dirs[j]
        # print(class1,'\n',class2)

        class1_name = os.path.basename(class1)
        class2_name = os.path.basename(class2)
        # 获取当前目录下所有的有效图片文件
        extensions = 'png'
        file_list1 = []
        file_list11 = []
        file_list2 = []
        file_list22 = []
        # 把图片存放在file_list列表里
        file_glob1 = os.path.join(INPUT_DATA, class1_name, '*.' + extension)
        file_list1.extend(glob.glob(file_glob1))
        file_list11.extend(glob.glob(file_glob1))
        file_list11.sort()
        file_glob2 = os.path.join(INPUT_DATA, class2_name, '*.' + extension)
        file_list2.extend(glob.glob(file_glob2))
        file_list22.extend(glob.glob(file_glob2))

        file_list22.sort()
        len_of_file = len(file_list1)
        if file_list1 and file_list2:
            b = np.arange(len_of_file)
            b_com = []
            for m, n in combinations(b, 2):
                # print(m,n)
                b_com.append([m, n])

                base_name1 = file_list11.index(file_list1[m % len(file_list1)]) + 1  # 获取文件的名称

                base_name2 = file_list22.index(file_list2[n % len(file_list2)]) + 1
                # unmatched_result.add([class1_name, base_name1, class2_name, base_name2])
                s = class1_name + '\t' + str(base_name1) + '\t' + class2_name + '\t' + str(base_name2)
                if s not in unmatched_result:
                    unmatched_result.add(
                        class1_name + '\t' + str(base_name1) + '\t' + class2_name + '\t' + str(base_name2))
                k = k + 1
    return unmatched_result, k


result, k1 = create_image_lists()
print(len(result))
# print(result)

result_un, k2 = create_pairs()
print(len(result_un))
# print(result_un)

file = open('/run/media/root/0ec8ef97-3812-4bf2-88a2-127cb286cf3e/Desktop/jk/jk/facenet-pytorch-main/fake_pairs.txt', 'w')

result1 = list(result)
result2 = list(result_un)

file.write('10 300\n')

j = 0
for i in range(10):
    j = 0
    print("=============================================第" + str(i) + '次, 相同的')
    for pair in result1[i * 200:i * 200 + 200]:
        j = j + 1
        print(str(j) + ': ' + pair)
        file.write(pair + '\n')

    print("=============================================第" + str(i) + '次, 不同的')
    for pair in result2[i * 200:i * 200 + 200]:
        j = j + 1
        print(str(j) + ': ' + pair)
        file.write(pair + '\n')
