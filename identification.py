# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 15:21:02 2019

@author: pozir
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train_ident = pd.read_csv('data/train_identity.csv')

train_ident['DeviceInfo'].value_counts()
list_colnames = []
list_nasum = []
for i in train_ident:
    na_num = train_ident[i].isna().sum()
#    print(na_num)
    list_nasum.append(na_num)
    list_colnames.append(i)
 