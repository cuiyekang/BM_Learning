import numpy as np

dataset_filename="./docs/python/t_data/affinity_dataset.txt"

x = np.loadtxt(dataset_filename)

print(x[:5])