import mat
import random
import time
import sys

a = [[1,2,3],
     [4,5,6]]
b = [[10,11],
     [20,21],
     [30,31]]
expected_c = [[140, 146],
              [320, 335]]
assert(mat.mul(a, b) == expected_c)
print("Basic sanity passed")

reps = 1000
rows_a = 100
cols_a = 50
rows_b = 50
cols_b = 200

data = []
for _ in range(reps):
    a = [[random.random() for _ in range(cols_a)] for _ in range(rows_b)]
    b = [[random.random() for _ in range(rows_b)] for _ in range(cols_a)]
    data.append([a,b])

start_t = time.time()
for i in range(len(data)):
    mat.mul(data[i][0], data[i][1])
print(f'Multiplied {len(data)} pairs of matrices in {time.time() - start_t:.3f}s')

sys.exit()
