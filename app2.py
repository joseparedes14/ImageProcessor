# ----- Libraries -----

from PIL import Image
import os
import sys
import time
import tracemalloc
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# ------ 


L = 256

# ---------- LOADING AND SAVING IMAGES ----------
def readimage(path):
    im = Image.open(path)
    arr = np.array(im.getdata())
    if arr.ndim == 1: 
        numColorChannels = 1
        arr = arr.reshape(im.size[1], im.size[0])
    else:
        numColorChannels = arr.shape[1]
        arr = arr.reshape(im.size[1], im.size[0], numColorChannels)
    return arr.astype(np.uint8)


def saveimage(arr, output=None):
    newIm = Image.fromarray(arr.astype(np.uint8))
    if output:
        newIm.save(output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"images_output/result-{timestamp}.bmp"
        newIm.save(default_name)


def parse_params(params):
    dic_param = dict()
    for param in params:
        values = param.strip('-').split('=')
        dic_param[values[0]] = values[1]
    return dic_param




# ------- BUILDING AND SAVING HISTOGRAMS -----------
def doHistogram(params):
    dic_param = parse_params(params)
    img = readimage(dic_param['input'])

    if img.ndim == 3:
        channel = int(dic_param.get('channel', 0))
        img = img[:, :, channel]

    histogram, bins = np.histogram(img, bins=256, range=(0, 256)) 
    histogram = histogram / (img.shape[0] * img.shape[1])

    if dic_param.get('image', False):
        plt.bar(range(256), histogram, width=1)
        plt.savefig(dic_param.get('output', 'images_output/histogram.png'))
        plt.close()

    return histogram




# -------- HISTOGRAM CALCULATION ALGORITHM --------- 
# H1

def doHUniform(params):
    histogram = doHistogram(params)
    dic_param = parse_params(params)
    img = readimage(dic_param['input'])

    s = np.zeros_like(histogram)

    minimum = int(dic_param.get('minimum', 0))
    maximum = int(dic_param.get('maximum', L-1))

    if not (0 <= minimum < maximum < L):
        raise ValueError("Invalid minimum or maximum values")

    for i in range(len(s)):
        s[i] = (maximum-minimum)*np.sum(histogram[0:i+1]) + minimum

    s = np.round(s).astype(np.uint8)

    img_enhanced = np.copy(img)

    if img.ndim == 3:
        channel = int(dic_param.get('channel', 0))
        img_enhanced[:, :, channel] = s[img[:, :, channel]]
    else:
        img_enhanced = s[img]

    output_path = dic_param.get('output', 'images_output/enhanced_uniform.png')
    saveimage(img_enhanced, output_path)

    return img_enhanced



# ---------- IMAGE CHARACTERISTICS ----------------
# C1
def cmean(params):
    p = doHistogram(params)
    m = np.arange(L)
    mean = np.sum(m * p)
    print(f"Mean: {mean}")
    return mean

def cvariance(params):
    p = doHistogram(params)
    m = np.arange(L)
    mean = np.sum(m * p)
    variance = np.sum((m-mean)**2*p)
    print(f"Variance: {variance}")
    return variance
    

# C2

def cstdev(params):
    p = doHistogram(params)
    m = np.arange(L)
    mean = np.sum(m * p)
    variance = np.sum((m-mean)**2*p)
    sigma = np.sqrt(variance)
    print(f'Standard deviation: {sigma}')
    return sigma


def cvarcoi(params):
    p = doHistogram(params)
    m = np.arange(L)
    mean = np.sum(m * p)
    variance = np.sum((m-mean)**2*p)
    sigma = np.sqrt(variance)
    if mean !=0:
        result = sigma/mean
    else:
        result = float('nan')
    print(f"Variation coefficient I: {result}")
    return result


# C3
def casyco(params):
    p = doHistogram(params)
    m = np.arange(L)

    mean = np.sum(m * p)
    variance = np.sum((m - mean) ** 2 * p)
    sigma = np.sqrt(variance)
    if sigma ** 3 > 0:
        result = np.sum((m - mean) ** 3 * p) / sigma ** 3
    else:
        result = float('nan')

    print(f"Asymmetry coefficient: {result}")
    return result

# C4
def cflaco(params):
    p = doHistogram(params)
    m = np.arange(L)

    mean = np.sum(m * p)
    variance = np.sum((m - mean) ** 2 * p)
    sigma = np.sqrt(variance)
    if sigma >0:
        result = np.sum((m - mean) ** 4 * p) / sigma ** 4 - 3
    else:
        result = float('nan')    
    print(f"Flattening coefficient: {result}")
    return result


# C5
def cvarcoii(params):
    p = doHistogram(params)
    result = np.sum(p ** 2)
    print(f"Variation coefficient II: {result}")
    return result


#C6
def centropy(params):
    p = doHistogram(params)
    p = p[p > 0]
    result = -np.sum(p * np.log2(p))
    print(f"Entropy: {result}")
    return result




# ------ Implementation of linear image filtration algorithm in spatial domain basing on convolution --------
# S4


S4_MASKS = {
    'S':  np.array([[-1,  1,  1],
                    [-1, -2,  1],
                    [-1,  1,  1]]),
    'SW': np.array([[ 1,  1,  1],
                    [-1, -2,  1],
                    [-1, -1,  1]]),
    'W':  np.array([[ 1,  1,  1],
                    [ 1, -2,  1],
                    [-1, -1, -1]]),
    'NW': np.array([[ 1,  1,  1],
                    [ 1, -2, -1],
                    [ 1, -1, -1]]),
}


# UNIVERSAL CONVOLUTION
def doConvolution(params):
    dic_param = parse_params(params)
    x = readimage(dic_param['input'])
    h = S4_MASKS[dic_param.get('mask', 'S')]

    M = h.shape[0] // 2
    P, Q = x.shape[0], x.shape[1]
    g = x.astype(float)

    for p in range(M, P - M):
        for q in range(M, Q - M):
            total = 0
            for i in range(-M, M + 1):
                for j in range(-M, M + 1):
                    total += h[i + M, j + M] * x[p + i, q + j]
            g[p, q] = total
    result = np.clip(np.round(g), 0, L - 1).astype(np.uint8)
    saveimage(result, dic_param.get('output'))
    return result



# OPTIMIZED CONVOLUTION
'''
def doSExDetII(params):
    dic_param = parse_params(params)
    x = readimage(dic_param['input']).astype(np.int32)

    P, Q = x.shape[0], x.shape[1]
    g = x.copy()

    r_prev = x[0, 0:Q-2] + x[0, 1:Q-1] + x[0, 2:Q]
    r_cur  = x[1, 0:Q-2] + x[1, 1:Q-1] + x[1, 2:Q]
    for p in range(1, P - 1):
        r_next = x[p+1, 0:Q-2] + x[p+1, 1:Q-1] + x[p+1, 2:Q]
        g[p, 1:Q-1] = r_prev + r_cur - r_next - 3 * x[p, 1:Q-1]
        r_prev = r_cur
        r_cur = r_next
    result = np.clip(g, 0, L - 1).astype(np.uint8)
    saveimage(result, dic_param.get('output'))
    return result
'''

def doSExDetII(params):
    dic_param = parse_params(params)
    x = readimage(dic_param['input']).astype(np.int32)   # int32: no uint8 wrap-around

    P, Q = x.shape[0], x.shape[1]
    g = x.copy()                                          # borders = original pixels

    for p in range(1, P - 1):
        cs = x[p-1] + x[p] + x[p+1]                       # vertical 3-sum of every column
        # S mask = right column + middle column - left column - 3*center
        g[p, 1:Q-1] = cs[2:Q] + cs[1:Q-1] - cs[0:Q-2] - 3 * x[p, 1:Q-1]

    result = np.clip(g, 0, L - 1).astype(np.uint8)
    saveimage(result, dic_param.get('output'))
    return result



# -------- Implement non-linear image filtration algorithm in the spatial domain -------------
# O3

def doSobel(params):
    dic_param = parse_params(params)
    imag = readimage(dic_param['input']).astype(np.int32)
    imag_new = imag.astype(np.float32) 

    height = imag.shape[0]
    width = imag.shape[1]

    # Offset para ignorar los bordes
    offset = 1

    for x in range(offset, height - offset):
        for y in range(offset, width - offset):

            
            A0 = imag[x-1, y-1]
            A1 = imag[x-1, y]
            A2 = imag[x-1, y+1]
            A3 = imag[x, y+1]
            A4 = imag[x+1, y+1]
            A5 = imag[x+1, y]
            A6 = imag[x+1, y-1]
            A7 = imag[x, y-1]

            
            X = (A2 + 2*A3 + A4) - (A0 + 2*A7 + A6)
            Y = (A0 + 2*A1 + A2) - (A6 + 2*A5 + A4)

            
            imag_new[x, y] = np.sqrt(X.astype(np.float32)**2 + Y.astype(np.float32)**2)

  
    imag_new = np.clip(imag_new, 0, 255).astype(np.uint8)

    saveimage(imag_new, dic_param.get('output'))
    return imag_new






if len(sys.argv) == 1:
    print("No command line parameters given.\n")
    sys.exit()

if len(sys.argv) == 2 and sys.argv[1] != '--help':
    print("Too few command line parameters given.\n")
    sys.exit()

command = sys.argv[1]
param = sys.argv[2:]
# .....






if command == '--histogram':
    doHistogram(param)

elif command == '--huniform':
    doHUniform(param)

elif command == '--cmean':
    cmean(param)
    
elif command == '--cvariance':
    cvariance(param)

elif command == '--cstdev':
    cstdev(param)
elif command == '--cvarcoi':
    cvarcoi(param)

elif command == '--casyco':
    casyco(param)

elif command == '--cflaco':
    cflaco(param)

elif command == '--cvarcoii':
    cvarcoii(param)

elif command == '--centropy':
    centropy(param)
elif command == '--convolution':
    doConvolution(param)

elif command == '--sexdetii':
    doSExDetII(param)


elif command == '--osobel':
    doSobel(param)