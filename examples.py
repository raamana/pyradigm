
import numpy as np
from mldataset import MLDataset

ds = MLDataset()
ds

data_dict = {}
for ix in range(10):
    data_dict[ix] = [np.random.random(1), ]

label_set = [1, 2, 2, 1, 1, 1, 2, 2, 1, 2]
lbl_dict = dict(zip(range(10), label_set))

ds2 = MLDataset(data_dict, lbl_dict)
ds2.description = 'Example dataset filled with random features and classes!'
print ds2


