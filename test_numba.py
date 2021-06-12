# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 16:43:16 2021

@author: Morais
"""

from numba import jit
import numpy as np
import time

N = 1000
x = np.arange(N * N).reshape(N, N)

@jit(nopython=True)
def go_fast_normal(a): # Function is compiled and runs in machine code
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace


start = time.time()
go_fast_normal(x)
end = time.time()
print("Elapsed (normal) = %s" % (end - start))

@jit(nopython=True)
def go_fast_jit(a): # Function is compiled and runs in machine code
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace

start = time.time()
go_fast_jit(x) # the first time, the function compiles
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
go_fast_jit(x)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
