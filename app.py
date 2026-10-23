# ----- Libraries -----

from PIL import Image
import sys
import numpy as np
from datetime import datetime


# ------ 


# ---- HELP ----

def show_help():
    print("""
IMAGE PROCESSOR - HELP
======================

Usage:
    python app.py --command -parameter=value ...

All image paths must point to valid image files.
For image processing operations, -input and -output are required.

--------------------------------------------------
BASIC IMAGE OPERATIONS
--------------------------------------------------

1. --brightness
   Modifies the brightness of an image by adding a constant
   value to every pixel.

   Required parameters:
       -input=PATH       Path to the input image.
       -output=PATH      Path to save the processed image.
       -value=INTEGER    Brightness adjustment.

   Values:
       Positive values make the image brighter.
       Negative values make the image darker.
       Zero leaves the image unchanged.
       Pixel values are clipped to the range [0, 255].

   Example:
       python app.py --brightness -input=images_input/lena.bmp -output=images_output/bright.bmp -value=30


2. --contrast
   Modifies image contrast using a linear transformation
   around the intensity value 127.

   Required parameters:
       -input=PATH       Path to the input image.
       -output=PATH      Path to save the processed image.
       -value=NUMBER     Contrast factor (alpha).

   Values:
       alpha > 1 increases contrast.
       0 < alpha < 1 decreases contrast.
       alpha = 1 preserves the original contrast.
       alpha = 0 produces a uniform mid-gray image.
       Negative values invert intensity differences around 127.

   Example:
       python app.py --contrast -input=images_input/lena.bmp -output=images_output/contrast.bmp -value=1.5


3. --negative
   Creates the negative of an image by subtracting each
   pixel value from 255.

   Required parameters:
       -input=PATH       Path to the input image.
       -output=PATH      Path to save the processed image.

   Example:
       python app.py --negative -input=images_input/lena.bmp -output=images_output/negative.bmp


--------------------------------------------------
GEOMETRIC TRANSFORMATIONS
--------------------------------------------------

4. --hflip
   Flips an image horizontally, reversing the order of
   its columns (left to right).

   Required parameters:
       -input=PATH       Path to the input image.
       -output=PATH      Path to save the processed image.

   Example:
       python app.py --hflip -input=images_input/lena.bmp -output=images_output/hflip.bmp


5. --vflip
   Flips an image vertically, reversing the order of
   its rows (top to bottom).

   Required parameters:
       -input=PATH       Path to the input image.
       -output=PATH      Path to save the processed image.

   Example:
       python app.py --vflip -input=images_input/lena.bmp -output=images_output/vflip.bmp


6. --dflip
   Reverses both the rows and columns of an image,
   producing a 180-degree rotation.

   Required parameters:
       -input=PATH       Path to the input image.
       -output=PATH      Path to save the processed image.

   Example:
       python app.py --dflip -input=images_input/lena.bmp -output=images_output/dflip.bmp


7. --shrink
   Reduces the image dimensions by an integer factor.
   It keeps one pixel from each factor-by-factor region,
   specifically the pixel at the top-left position.

   Required parameters:
       -input=PATH       Path to the input image.
       -output=PATH      Path to save the processed image.
       -value=INTEGER    Shrinking factor (greater than zero).

   Example:
       python app.py --shrink -input=images_input/lena.bmp -output=images_output/small.bmp -value=2


8. --enlarge
   Enlarges an image by an integer factor using nearest-neighbor
   replication. Each original pixel is repeated horizontally
   and vertically.

   Required parameters:
       -input=PATH       Path to the input image.
       -output=PATH      Path to save the processed image.
       -value=INTEGER    Enlargement factor (greater than zero).

   Example:
       python app.py --enlarge -input=images_input/lena.bmp -output=images_output/large.bmp -value=2


--------------------------------------------------
SPATIAL FILTERS
--------------------------------------------------

9. --amean
   Applies an arithmetic mean filter to reduce local variations
   and smooth the image. Each pixel is replaced by the mean
   of the pixel values within its local window.

   Required parameters:
       -input=PATH       Path to the input image.
       -output=PATH      Path to save the filtered image.

   Optional parameters:
       -m=INTEGER        Window height (default: 3).
       -n=INTEGER        Window width (default: 3).

   Note:
       The default window size is 3x3.
       Border pixels outside the processed region remain unchanged.

   Example:
       python app.py --amean -input=images_input/lena.bmp -output=images_output/amean.bmp -m=3 -n=3


--------------------------------------------------
IMAGE QUALITY METRICS
--------------------------------------------------

10. --mse
    Calculates the Mean Squared Error between two images.

    Required parameters:
        -input=PATH      Path to the original image.
        -compared=PATH   Path to the image being compared.

    Output:
        Prints the MSE value to the terminal.
        Lower values indicate smaller average squared differences.
        An MSE of 0 means the images are identical.

    Example:
        python app.py --mse -input=images_input/lena.bmp -compared=images_output/negative.bmp


11. --pmse
    Calculates the Peak-normalized Mean Squared Error by
    dividing the MSE by 255 squared.

    Required parameters:
        -input=PATH      Path to the original image.
        -compared=PATH   Path to the image being compared.

    Output:
        Prints the PMSE value to the terminal.
        A value of 0 means the images are identical.

    Example:
        python app.py --pmse -input=images_input/lena.bmp -compared=images_output/negative.bmp


12. --md
    Calculates the Maximum Difference between two images,
    defined as the largest absolute pixel-value difference.

    Required parameters:
        -input=PATH      Path to the original image.
        -compared=PATH   Path to the image being compared.

    Output:
        Prints the maximum absolute difference to the terminal.
        A value of 0 means the images are identical.

    Example:
        python app.py --md -input=images_input/lena.bmp -compared=images_output/negative.bmp


13. --psnr
    Calculates the Peak Signal-to-Noise Ratio between two images in decibels (dB).

    Required parameters:
        -input=PATH      Path to the original image.
        -compared=PATH   Path to the image being compared.

    Output:
        Prints the PSNR value in dB to the terminal.
        Higher values generally indicate higher image quality and closer similarity.

    Example:
        python app.py --psnr -input=images_input/lena.bmp -compared=images_output/negative.bmp

--------------------------------------------------
GENERAL NOTES
--------------------------------------------------

- Use quotation marks around paths containing spaces.
- Input images must be readable by the program.
- Image processing operations save their results to the
  specified output path.
- Image quality metrics print numerical results instead
  of saving a new image.
- Pixel intensity values are represented in the range [0, 255].
- The program supports grayscale and color images for the
  operations implemented in the source code.

""")



# ----- Code ------


# Basic Functions


def readimage(path):
    im = Image.open(path)
    
    
    arr = np.array(im.getdata())
    # .getdata() devuelve los datos como una larga tira de pixeles sin 
    # organizar en filas y columnas.
    # Si es en color, arr = [ (255, 0, 0), (100, 150, 200), ... ]
    # Si es en blanco y negro, arr = [ 0, 1, 4, 2, 132, ...]
    
    if arr.ndim == 1: 
        # si cada posicion del array tiene un solo valor, es en escala de grises
        # ese valor es un entero de 8 bits que representa un valor entre 0-255
        # osea que cada posicion del array guarda un nº entre 0-255, siendo 0 negro y 255 blanco
        numColorChannels = 1
        arr = arr.reshape(im.size[1], im.size[0])
        # im.size[0] es el ancho, nº columnas
        # im.size[1] es el alto, nº filas
        # como la imagen es una cuadrícula y no una fila larga,
        # hacemos reshape para recortar y ordenar en forma de tabla
    
    else:
        numColorChannels = arr.shape[1]

        # en una imágen a color, cada píxel tiene 3 números, por eso
        # la tira plana debe convertirse en una tabla de 3 dimensiones.
        # En un cubo: alto, ancho y los 3 canales de color.
        arr = arr.reshape(im.size[1], im.size[0], numColorChannels)

    return arr


def saveimage(arr, output=None):

    newIm = Image.fromarray(arr.astype(np.uint8))

    if output:
        newIm.save(output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # to avoid using : on names, because on Windows is 
        # prohibited.
        default_name = f"result-{timestamp}.bmp"
        newIm.save(default_name)


def parse_params(params):
    dic_param = dict()
    for param in params:
        values = param.strip('-').split('=')
        dic_param[values[0]] = values[1]
    return dic_param


# ------- Elementary operations ----------


#(B1) Image brightness modification (--brightness)

def doBrightness(params):
    dic_param = parse_params(params)

    imag = readimage(dic_param['input'])

    constant = int(dic_param['value'])
    # es cuánto vamos a subir/bajar el brillo de todos los píxeles
    # a la vez

    if constant <= 0:
        imag = np.maximum(0, imag + constant)
        # * como nungun pixel puede bajar de 0, si el cálculo da p.e
        # -10, pues el maximo será 0
        # * al hacer img+constant, se le suma constant a cada elemento
        # de la matriz (a cada pixel)
    else:
        imag = np.minimum(255, imag + constant)

    saveimage(imag, dic_param['output'])


# (B2) Image contrast modification (--contrast)
def doContrast(params):
    # lineal version
    dic_param = parse_params(params)
    imag = readimage(dic_param['input'])
    alpha = float(dic_param['value'])
    center = 127
    result = np.minimum(255, np.maximum(0, center + alpha * (imag - center)))
    result = result.astype(np.uint8)
    saveimage(result, dic_param['output'])


# (B3) Negative (--negative)
def doNegative(params):
    dic_param = parse_params(params)
    imag = readimage(dic_param['input'])
    result = 255 - imag
    saveimage(result, dic_param['output'])
    


# ------- Geometric transforms ----------
   
# (G1) Horizontal flip (--hflip)    
def doHorFlip(params):
    dic_param = parse_params(params)
    imag = readimage(dic_param['input'])
    result = imag[:, ::-1]
    saveimage(result, dic_param['output'])
    


# (G2) Vertical flip (--vflip)
def doVerFlip(params):
    dic_param = parse_params(params)
    imag = readimage(dic_param['input'])
    result = imag[::-1]
    saveimage(result, dic_param['output'])



# (G3) Diagonal flip (--dflip)
def doDiagonalFlip(params):
    dic_param = parse_params(params)
    imag = readimage(dic_param['input'])
    result = imag[::-1, ::-1]
    saveimage(result, dic_param['output'])
 


# (G4) Image shrinking (--shrink)
def doShrink(params):
    dic_param = parse_params(params)
    imag = readimage(dic_param['input'])
    factor = int(dic_param['value'])
    zeros = imag[::factor, ::factor]
    saveimage(zeros, dic_param['output'])
       
    

# (G5) Image enlargement (--enlarge)
def doEnlargement(params):
    dic_param = parse_params(params)
    imag = readimage(dic_param['input'])
    factor = int(dic_param['value'])
    zeros = np.repeat(np.repeat(imag, factor, axis=0), factor, axis=1)
    saveimage(zeros, dic_param['output'])



def doAMF(params):
    dic_param = parse_params(params)
    imag = readimage(dic_param['input'])
    # Obtenemos m y n de los parámetros (por defecto 3x3 si no se indican)
    m = int(dic_param.get('m', 3))
    n = int(dic_param.get('n', 3))
    imag_new = np.copy(imag)
    height = imag.shape[0]
    width = imag.shape[1]
    # Calculamos el radio o margen a ignorar en los bordes según el tamaño de la ventana
    offset_m = m // 2
    offset_n = n // 2
    
    for x in range(offset_m, height - offset_m):
        for y in range(offset_n, width - offset_n):
            if imag.ndim == 3:
                window = imag[x - offset_m : x + offset_m + 1, y - offset_n : y + offset_n + 1, :]
                imag_new[x, y] =  np.mean(window,axis=(0,1))
            else:
                window = imag[x - offset_m : x + offset_m + 1, y - offset_n : y + offset_n + 1]
                imag_new[x, y] =  np.mean(window)
    imag_new = imag_new.astype(np.uint8)
    saveimage(imag_new, dic_param['output'])




def doMidpoint(params):
    dic_param = parse_params(params)
    imag = readimage(dic_param['input'])
    # Obtenemos m y n de los parámetros (por defecto 3x3 si no se indican)
    m = int(dic_param.get('m', 3))
    n = int(dic_param.get('n', 3))
    imag_new = np.copy(imag)
    height = imag.shape[0]
    width = imag.shape[1]
    # Calculamos el radio o margen a ignorar en los bordes según el tamaño de la ventana
    offset_m = m // 2
    offset_n = n // 2
    
    for x in range(offset_m, height - offset_m):
        for y in range(offset_n, width - offset_n):
            if imag.ndim == 3:
                window = imag[x - offset_m : x + offset_m + 1, y - offset_n : y + offset_n + 1, :]
                vmin = np.min(window, axis=(0, 1))
                vmax = np.max(window, axis=(0, 1))      
            else:
                window = imag[x - offset_m : x + offset_m + 1, y - offset_n : y + offset_n + 1]
                vmin = np.min(window)
                vmax = np.max(window)  
            imag_new[x, y] = (vmax.astype(np.float32) + vmin.astype(np.float32)) / 2            
    imag_new = imag_new.astype(np.uint8)
    saveimage(imag_new, dic_param['output'])
    

# ------- Geometric transforms ----------

# (E1) Mean square error (--mse)
def doMSE(params):
    dic_param = parse_params(params)
    img = readimage(dic_param['input'])
    img_res = readimage(dic_param['compared'])
    
    img1_f = img.astype(np.float32)
    img2_f = img_res.astype(np.float32)
    mse_value = np.mean((img1_f - img2_f) ** 2)
    print(f"MSE is: {mse_value:.4f}")
    
def doPMSE(params):
    dic_param = parse_params(params)
    img = readimage(dic_param['input'])
    img_res = readimage(dic_param['compared'])
    img1_f = img.astype(np.float32)
    img2_f = img_res.astype(np.float32)
    mse_value = np.mean((img1_f - img2_f) ** 2)
    if mse_value == 0:
        pmse_value = 0.0
    else:
        pmse_value = mse_value / (255 ** 2)
    print(f"PMSE is: {pmse_value:.4f}")
        
        
def doMD(params):
    dic_param = parse_params(params)
    img = readimage(dic_param['input'])
    img_res = readimage(dic_param['compared'])

    img1_f = img.astype(np.float32)
    img2_f = img_res.astype(np.float32)

    md_value = np.max(np.abs(img1_f - img2_f))

    print(f"MD is: {md_value:.4f}")        
    
        

# Signal-to-noise ratio (--snr)
def doSNR(params):
    dic_param = parse_params(params)
    img = readimage(dic_param['input'])
    img_res = readimage(dic_param['compared'])

    img1_f = img.astype(np.float32)
    img2_f = img_res.astype(np.float32)

    signal_power = np.sum(img1_f ** 2)
    noise_power = np.sum((img1_f - img2_f) ** 2)

    if noise_power == 0:
        snr_value = float('inf')
    elif signal_power == 0:
        snr_value = float('-inf')
    else:
        snr_value = 10 * np.log10(signal_power / noise_power)

    print(f"SNR is: {snr_value:.4f} dB")


# Peak signal-to-noise ratio (--psnr)
def doPSNR(params):
    dic_param = parse_params(params)
    img = readimage(dic_param['input'])
    img_res = readimage(dic_param['compared'])

    img1_f = img.astype(np.float32)
    img2_f = img_res.astype(np.float32)

    noise_power = np.sum((img1_f - img2_f) ** 2)
    peak_value = np.max(img1_f)

    if noise_power == 0:
        psnr_value = float('inf')
    elif peak_value == 0:
        psnr_value = float('-inf')
    else:
        psnr_value = 10 * np.log10(
            (img1_f.size * peak_value ** 2) / noise_power
        )

    print(f"PSNR is: {psnr_value:.4f} dB")
    
    




if len(sys.argv) == 1:
    print("No command line parameters given.\n")
    sys.exit()

if len(sys.argv) == 2 and sys.argv[1] != '--help':
    print("Too few command line parameters given.\n")
    sys.exit()

command = sys.argv[1]
param = sys.argv[2:]
# .....






if command == '--brightness':
    doBrightness(param)

elif command == '--hflip':
    doHorFlip(param)

elif command == '--vflip':
    doVerFlip(param)

elif command == '--dflip':
    doDiagonalFlip(param)

elif command == '--contrast':
    doContrast(param)

elif command == '--negative':
    doNegative(param)

elif command == '--shrink':
    doShrink(param)

elif command == '--enlarge':
    doEnlargement(param)

elif command == '--mse':
    doMSE(param)

elif command == '--pmse':
    doPMSE(param)
    
elif command == '--md':
    doMD(param)


elif command == '--amean':
    doAMF(param)


elif command == '--help':
    show_help()

elif command == '--psnr':
    doPSNR(param)

elif command == '--snr':
    doSNR(param)

elif command == '--psnr':
    doPSNR(param)