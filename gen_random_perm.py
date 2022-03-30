'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import numpy as np
import os
import glob


def write_file_perm(file_hdl, perm, data):
    for idx in perm:
        file_hdl.write(data[idx])
    file_hdl.close()


filepath = './data/fashion/fashion_mnist_train_list.txt'
f = open(filepath).readlines()
num_data = len(f)

rand_1 = np.random.RandomState(seed=2018).permutation(num_data)
rand_2 = np.random.RandomState(seed=2019).permutation(num_data)
rand_3 = np.random.RandomState(seed=2020).permutation(num_data)
rand_4 = np.random.RandomState(seed=2021).permutation(num_data)
rand_5 = np.random.RandomState(seed=2022).permutation(num_data)

rand_all = [rand_1, rand_2, rand_3, rand_4, rand_5]

for i, rand_n in enumerate(rand_all):
    file_rand_n = open('./data/fashion/%s_%d.txt' % (domain, i), 'w')
    write_file_perm(file_rand_n, rand_n, f)




