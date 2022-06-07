from re import search
import numpy as np
from numpy import linalg as npla
import cv2
import matplotlib.pyplot as plt
import imageio
from scipy.optimize import linprog
from PIL import Image

# Reading images
x=imageio.imread('data/city100.png')
trg = np.float32(x)
y=imageio.imread('data/starry100.png')
src = np.float32(y)

# Checking data type of input
if not np.issubdtype(src.dtype, np.floating):
    raise ValueError("src value must be float")
if not np.issubdtype(trg.dtype, np.floating):
    raise ValueError("trg value must be float")
# Checking if image has 3 channels
if len(src.shape) != 3:
    raise ValueError("src shape must have rank 3 (h,w,c)")
# Checking if source and target images have equal dimensions
if src.shape != trg.shape:
    raise ValueError("src and trg shapes must be equal")
    
src_dtype = src.dtype        
h,w,c = src.shape
new_src = src.copy()

# Resizing process
advect = np.zeros ( (h*w,c), dtype=src_dtype )
dir = np.random.normal(size=c).astype(src_dtype)
dir /= npla.norm(dir)

P_r = np.sum( new_src*dir, axis=-1).reshape ((h*w))
P_t = np.sum( trg*dir, axis=-1).reshape ((h*w))

P_r = P_r / np.sum(P_r) # Source distribution
P_t = P_t / np.sum(P_t) # Target distribution
l=len(P_r);
D = np.ndarray(shape=(l, l))

for i in range(l):
  for j in range(l):
    D[i,j] = (abs(range(l)[i] - range(l)[j]))

A_r = np.zeros((l, l, l))
A_t = np.zeros((l, l, l))

for i in range(l):
  for j in range(l):
    A_r[i, i, j] = 1
    A_t[i, j, i] = 1

# Generating A matrix
A = np.concatenate((A_r.reshape((l, l**2)), A_t.reshape((l, l**2))), axis=0)
print("A: \n", A, "\n")

b = np.concatenate((P_r, P_t), axis=0)
c = D.reshape((l**2))

from scipy.optimize import linprog
from matplotlib import cm

#Using solver

opt_res = linprog(c, A_eq=A, b_eq=b, bounds=[0, None])
emd = opt_res.fun
gamma = opt_res.x.reshape((l, l))
print("EMD: ", emd, "\n")

plt.imshow(gamma, cmap=cm.gist_heat, interpolation='nearest')
plt.axis('off')
plt.savefig("transport_plan.svg")
print("Gamma:")
plt.show()

plt.imshow(D, cmap=cm.gist_heat, interpolation='nearest')
plt.axis('off')
plt.savefig("distances.svg")
print("D:")
plt.show()

# Using dual
opt_res = linprog(-b, A.T, c, bounds=(None, None))

emd = -opt_res.fun
f = opt_res.x[0:l]
g = opt_res.x[l:]

print("dual EMD: ", emd)

print("f: \n", f)
plt.plot(range(l), f)
plt.savefig("f_function.svg")
plt.show()


print("g: \n", f)
plt.plot(range(l), g)
plt.showopt_res = linprog(c, A_eq=A, b_eq=b, bounds=[0, None])
