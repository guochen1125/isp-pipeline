import numpy as np

def yuyv2bggr(yuyv):
    yuyv = yuyv.astype(np.uint16)
    bggr = ((yuyv[1:: 2] << 8) + yuyv[:: 2]).flatten()
    return bggr


