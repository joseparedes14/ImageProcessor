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

def show_help():
    print("""
IMAGE PROCESSOR - HELP
======================

Usage:
    python app2.py --command -argument=value [...]

All image paths must point to valid image files.
For all operations, -input is required.
If -output is not given, the result is saved in images_output/
with a timestamped name (images_output/ must exist).

--------------------------------------------------
HISTOGRAMS (H1)
--------------------------------------------------

1. --histogram
   Computes the normalized histogram of an image (256 bins,
   each value divided by the total number of pixels).
   Optionally saves it as a bar chart.

   Required parameters:
       -input=PATH       Path to the input image.

   Optional parameters:
       -channel=INTEGER  Color channel to analyze (default: 0).
                         Ignored for grayscale images.
       -image=1          If present, saves the histogram as a plot.
       -output=PATH      Path to save the plot
                         (default: images_output/histogram.png).

   Notes:
       -output is only used when -image is given.

   Example:
       python app2.py --histogram -input=images_input/lena.bmp -image=1 -output=images_output/hist.png


2. --huniform
   Enhances an image using the uniform histogram modification
   (histogram equalization), remapping brightness levels so that
   they spread between a minimum and a maximum value.

   Required parameters:
       -input=PATH       Path to the input image.

   Optional parameters:
       -minimum=INTEGER  Minimum output brightness (default: 0).
       -maximum=INTEGER  Maximum output brightness (default: 255).
       -channel=INTEGER  Color channel to transform (default: 0).
                         Ignored for grayscale images.
       -output=PATH      Path to save the processed image
                         (default: images_output/enhanced_uniform.png).

   Values:
       0 <= minimum < maximum <= 255, otherwise an error is raised.

   Notes:
       In color images, only the selected channel is modified.

   Example:
       python app2.py --huniform -input=images_input/lena.bmp -output=images_output/huniform.bmp -minimum=0 -maximum=255


--------------------------------------------------
IMAGE CHARACTERISTICS (C1 - C6)
--------------------------------------------------

All characteristics are computed from the normalized histogram
and printed on screen. No output image is produced.

   Required parameters (for all of them):
       -input=PATH       Path to the input image.

   Optional parameters (for all of them):
       -channel=INTEGER  Color channel to analyze (default: 0).
                         Ignored for grayscale images.

3. --cmean
   Calculates the mean brightness of the image.

   Example:
       python app2.py --cmean -input=images_input/lena.bmp


4. --cvariance
   Calculates the variance of the brightness levels.

   Example:
       python app2.py --cvariance -input=images_input/lena.bmp


5. --cstdev
   Calculates the standard deviation (square root of the variance).

   Example:
       python app2.py --cstdev -input=images_input/lena.bmp


6. --cvarcoi
   Calculates the variation coefficient I (standard deviation
   divided by the mean).

   Notes:
       Returns NaN if the mean is zero.

   Example:
       python app2.py --cvarcoi -input=images_input/lena.bmp


7. --casyco
   Calculates the asymmetry coefficient (skewness) of the histogram.

   Notes:
       Returns NaN if the standard deviation is zero.

   Example:
       python app2.py --casyco -input=images_input/lena.bmp


8. --cflaco
   Calculates the flattening coefficient (excess kurtosis)
   of the histogram.

   Notes:
       Returns NaN if the standard deviation is zero.

   Example:
       python app2.py --cflaco -input=images_input/lena.bmp


9. --cvarcoii
   Calculates the variation coefficient II (sum of the squared
   histogram values).

   Example:
       python app2.py --cvarcoii -input=images_input/lena.bmp


10. --centropy
    Calculates the information source entropy of the image,
    in bits.

    Example:
        python app2.py --centropy -input=images_input/lena.bmp


--------------------------------------------------
LINEAR FILTERING - CONVOLUTION (S4)
--------------------------------------------------

11. --convolution
    Applies a 3x3 convolution mask to the image (universal
    implementation, slow). The masks are directional edge
    detectors.

    Required parameters:
        -input=PATH       Path to the input image.

    Optional parameters:
        -mask=NAME        Mask to apply: S, SW, W or NW (default: S).
        -output=PATH      Path to save the processed image.
        
    Example:
        python app2.py --convolution -input=images_input/lena.bmp -output=images_output/conv.bmp -mask=SW

12. --sexdetii
    Optimized version of the convolution with mask S. It uses
    row sums and vectorized operations, producing the same
    result as --convolution much faster.

    Required parameters:
        -input=PATH       Path to the input image.

    Optional parameters:
        -output=PATH      Path to save the processed image.

    Example:
        python app2.py --sexdetii -input=images_input/lena.bmp -output=images_output/sexdetii.bmp

--------------------------------------------------
NON-LINEAR FILTERING - SOBEL OPERATOR (O3)
--------------------------------------------------

13. --osobel
    Applies the Sobel operator for edge detection. For each pixel,
    the horizontal and vertical gradients are computed and combined
    as sqrt(X^2 + Y^2).

    Required parameters:
        -input=PATH       Path to the input image.

    Optional parameters:
        -output=PATH      Path to save the processed image.

    Example:
        python app2.py --osobel -input=images_input/lena.bmp -output=images_output/sobel.bmp


--------------------------------------------------
EXECUTION TIME
--------------------------------------------------

14. --benchmark
    Compares the execution time of --convolution (mask S) and
    --sexdetii on one or several images, and prints a table
    with the times and the speedup.

    Required parameters:
        -input=PATH[,PATH,...]  One or more image paths,
                                separated by commas (no spaces).

    Example:
        python app2.py --benchmark -input=images_input/lena.bmp,images_input/boat.bmp


""")

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
    # devuelve el vector de 256 valores, dividido entre el nº de píxeles
    # entonces histogram[m] = H(m)/N
    
    dic_param = parse_params(params)
    img = readimage(dic_param['input'])

    s = np.zeros_like(histogram)
    # es la tabla de transformación.
    # creamos un vector de 256 ceros, s[f] será el brillo 
    # nuevo para el brillo antiguo f

    minimum = int(dic_param.get('minimum', 0))
    maximum = int(dic_param.get('maximum', L-1)) #L-1=255
    # los convierte a int porque el parseparam los devuelve str

    if not (0 <= minimum < maximum < L):
        raise ValueError("Invalid minimum or maximum values")

    # cálculo de la tabla
    for i in range(len(s)):
        s[i] = (maximum-minimum)*np.sum(histogram[0:i+1]) + minimum
        # histogram[0:i+1] son los valores H(0)/N ... H(i)/N
    s = np.round(s).astype(np.uint8)
    # redondea al entero mas cercano y convierte e uint8,
    # porque los brillos son enteros de 0 a 255

    img_enhanced = np.copy(img)

    if img.ndim == 3:
        channel = int(dic_param.get('channel', 0))
        img_enhanced[:, :, channel] = s[img[:, :, channel]]
        #aplicamos a toda la imágen de gole.
        # si es en color, solo se transforma el channel elegido
        # s[img] sustituye cada pixel por s[valor_del_pixel]
    else:
        img_enhanced = s[img]
        # transforma toda la imagen

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
    'S':  np.array([[-1, -1, -1],
                    [ 1, -2,  1],
                    [ 1,  1,  1]]),
    'SW': np.array([[ 1, -1, -1],
                    [ 1, -2, -1],
                    [ 1,  1,  1]]),
    'W':  np.array([[ 1,  1, -1],
                    [ 1, -2, -1],
                    [ 1,  1, -1]]),
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
                    total += h[M - i, M - j] * x[p + i, q + j]
            g[p, q] = total
    result = np.clip(np.round(g), 0, L - 1).astype(np.uint8)
    saveimage(result, dic_param.get('output'))
    return result



# OPTIMIZED CONVOLUTION


def doSExDetII(params):
    dic_param = parse_params(params)
    x = readimage(dic_param['input']).astype(np.int32)   # int32: no uint8 wrap-around

    P, Q = x.shape[0], x.shape[1]
    g = x.copy()                                          # borders = original pixels
    rs = x[:, 0:Q-2] + x[:, 1:Q-1] + x[:, 2:Q]
    g[1:P-1, 1:Q-1] = rs[0:P-2] - rs[2:P] + rs[1:P-1] - 3 * x[1:P-1, 1:Q-1]

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


# --- Time ---

def doBenchmark(params):
    dic_param = parse_params(params)
    print("| Image (size) | doConvolution [s] | doSExDetII [s] | Speedup |")
    print("|---|---|---|---|")
    for path in dic_param['input'].split(','):
        h, w = readimage(path).shape[:2]

        t0 = time.perf_counter()
        doConvolution([f'--input={path}', '--mask=S', '--output=images_output/bench1.bmp'])
        t_conv = time.perf_counter() - t0

        t0 = time.perf_counter()
        doSExDetII([f'--input={path}', '--output=images_output/bench2.bmp'])
        t_opt = time.perf_counter() - t0

        print(f"| {os.path.basename(path)} ({w}x{h}) | {t_conv:.3f} | {t_opt:.3f} | {t_conv / t_opt:.1f}x |")

# ---- Parameters -----


if len(sys.argv) == 1:
    print("No command line parameters given.\n")
    sys.exit()

if sys.argv[1] == '--help':
    show_help()
    sys.exit()

if len(sys.argv) == 2 and sys.argv[1] != '--help':
    print("Too few command line parameters given.\n")
    sys.exit()

command = sys.argv[1]
param = sys.argv[2:]




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
    
elif command == '--benchmark':
    doBenchmark(param)