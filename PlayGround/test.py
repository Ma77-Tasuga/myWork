import os
from Ours.scripts.config import *

# print(os.path.abspath('../Ours'))

# print(type(device))
#
# file = '../Ours/data/DARPA_T3/label/trace/feature.txt'
#
# with open(file, 'a', encoding='utf-8') as f:
#     f.write('\nEVENT_UPDATE' + '\t' + '23')

list1 = [1,2,3]
for item in list1:
    item +=2

print(list1)
set1 = frozenset((1,'abc',3))
set2 = frozenset(('abc',3,1))
dec = dict()
dec[set1] = 2

print(set1==set2)
print(set1)
print(set2)
print(dec[set2])