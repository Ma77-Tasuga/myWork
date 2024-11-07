import os
from Ours.scripts.config import *

# print(os.path.abspath('../Ours'))

# print(type(device))

file = '../Ours/data/DARPA_T3/label/trace/feature.txt'

with open(file, 'a', encoding='utf-8') as f:
    f.write('\nEVENT_UPDATE' + '\t' + '23')
