import numpy as np

for seed in range(10):
    np.random.seed(seed)
    task_order = np.random.permutation(10)
    print(task_order)