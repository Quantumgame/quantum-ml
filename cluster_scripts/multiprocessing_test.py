import multiprocessing as mpl
import numpy as np
import time

def foo():
    pass
n_cpu = mpl.cpu_count()
st = time.time()
with mpl.Pool(n_cpu) as p:
    out = p.map(np.sqrt,np.arange(10000))
print(time.time() - st)
