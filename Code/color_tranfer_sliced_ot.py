import numpy as np
import imageio
from numpy import linalg as npla
import cv2
import matplotlib.pyplot as plt
import imageio

def color_transfer(src,trg, steps=10, batch_size=5, reg_sigmaXY=16.0, reg_sigmaV=5.0):
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

    for step in range (steps):
        advect = np.zeros ( (h*w,c), dtype=src_dtype )# initializing advect vector
        for batch in range (batch_size):
            dir = np.random.normal(size=c).astype(src_dtype)    #drawing random samples
            dir /= npla.norm(dir)   #normalizing
            # Projections
            projsource = np.sum( new_src*dir, axis=-1).reshape ((h*w))
            projtarget = np.sum( trg*dir, axis=-1).reshape ((h*w))

            #Sorting
            idSource = np.argsort(projsource)
            idTarget = np.argsort(projtarget)

            # Generating difference
            a = projtarget[idTarget]-projsource[idSource]
            
            for i_c in range(c):
                advect[idSource,i_c] += a * dir[i_c]
        new_src += advect.reshape( (h,w,c) ) / batch_size
        
    # Using bilateral filter
    if reg_sigmaXY != 0.0:
        src_diff = new_src-src
        new_src = src + cv2.bilateralFilter (src_diff, 0, reg_sigmaV, reg_sigmaXY )
    return new_src


# Display routines
x=imageio.imread('data/city.png') # Target image
x = np.float32(x)
y=imageio.imread('data/starry.png')   # Source image
y = np.float32(y)
z=color_transfer(x,y)

fig = plt.figure()
fig.add_subplot(1, 3, 1)
plt.imshow(y/255)
plt.axis('off')
plt.title("Source Image")

fig.add_subplot(1, 3, 2)
plt.imshow(x/255)
plt.axis('off')
plt.title("Target Image")

fig.add_subplot(1, 3, 3)
plt.imshow(z/255)
plt.axis('off')
plt.title("Result")

# Histogram displays
colors = ("red", "green", "blue")
channel_ids = (0, 1, 2)

fig1=plt.figure()
fig1.add_subplot(1, 3, 1)
plt.xlim([0, 256])
for channel_id, c in zip(channel_ids, colors):
    histogram, bin_edges = np.histogram(
        y[:, :, channel_id], bins=256, range=(0, 256)
    )
    plt.plot(bin_edges[0:-1], histogram, color=c)
plt.show()
plt.title("Color Histogram of Source")
plt.xlabel("Color value")
plt.ylabel("Pixel count")


fig1.add_subplot(1, 3, 2)
plt.xlim([0, 256])
for channel_id, c in zip(channel_ids, colors):
    histogram1, bin_edges = np.histogram(
        x[:, :, channel_id], bins=256, range=(0, 256)
    )
    plt.plot(bin_edges[0:-1], histogram1, color=c)
plt.show()
plt.title("Color Histogram of Target")
plt.xlabel("Color value")
plt.ylabel("Pixel count")


fig1.add_subplot(1, 3, 3)
plt.xlim([0, 256])
for channel_id, c in zip(channel_ids, colors):
    histogram2, bin_edges = np.histogram(
        z[:, :, channel_id], bins=256, range=(0, 256)
    )
    plt.plot(bin_edges[0:-1], histogram2, color=c)
plt.show()
plt.title("Color Histogram of Result")
plt.xlabel("Color value")
plt.ylabel("Pixel count")

