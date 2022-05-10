import numpy as np
import time as t

start = t.time()

A = np.random.rand(3000, 3000)
b = np.random.rand(3000)
x = np.random.rand(3000)

r = b - np.dot(A, x)
p = np.copy(r)
rsold = np.dot(r, r)

for i in range(100):
    Ap = np.dot(A, p)
    alpha = rsold / np.dot(p, Ap)

    x = alpha * p + x
    r = r - alpha * Ap

    rsnew = np.dot(r, r)

    p = (rsnew / rsold) * p + r
    rsold = rsnew

end = t.time()

print(f'Executed in {end-start}s')