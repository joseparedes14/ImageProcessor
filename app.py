# ----- Libraries -----

from PIL import Image
import sys
import numpy as np
from datetime import datetime


# ------


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
    
    '''
    dic_param = parse_params(params)
    imag = readimage(dic_param['input'])

    zeros = np.zeros_like(imag)

    height = zeros.shape[0]
    width = zeros.shape[1]

    for x in range(height):
        for  y in range(width):
            zeros[x, width - y - 1] = imag[x,y]
 
    saveimage(zeros, dic_param['output'])
    '''


# (G2) Vertical flip (--vflip)
def doVerFlip(params):
    dic_param = parse_params(params)
    imag = readimage(dic_param['input'])
    result = imag[::-1]
    saveimage(result, dic_param['output'])
    
    '''
    zeros = np.zeros_like(imag)
    i = 0
    j = 0
    k = imag.shape[0] - 1 
    while k >=  0:
        while j <= imag.shape[1] - 1:
            zeros[i][j] = imag[k][j]
            j+=1
        j = 0
        i+=1
        k -= 1
    saveimage(zeros, dic_param['output'])
    '''


# (G3) Diagonal flip (--dflip)
def doDiagonalFlip(params):
    dic_param = parse_params(params)
    imag = readimage(dic_param['input'])
    result = imag[::-1, ::-1]
    saveimage(result, dic_param['output'])
    '''
    zeros = np.zeros_like(imag)
    i = 0
    k = imag.shape[0] - 1
    while k >= 0:
        j = imag.shape[1] - 1
        l = 0
        while j >= 0:
            zeros[i][l] = imag[k][j]
            j -= 1
            l += 1
        i += 1
        k -= 1
    '''


# (G4) Image shrinking (--shrink)
def doShrink(params):
    dic_param = parse_params(params)
    imag = readimage(dic_param['input'])
    factor = int(dic_param['value'])
    height = imag.shape[0]
    width = imag.shape[1]
    new_height = height // factor
    new_width = width // factor
    
    if imag.ndim == 3:  # Si es a color
        channels = imag.shape[2]
        zeros = np.zeros((new_height, new_width, channels), dtype=imag.dtype)
    else:  # Si es blanco y negro 
        zeros = np.zeros((new_height, new_width), dtype=imag.dtype)
           
    for i in range(new_height):
        for j in range(new_width):
            x = i * factor
            y = j * factor
            zeros[i,j] = imag[x,y]
    saveimage(zeros, dic_param['output'])
       
    

# (G5) Image enlargement (--enlarge)
def doEnlargement(params):
    dic_param = parse_params(params)
    imag = readimage(dic_param['input'])
    
    factor = int(dic_param['value'])
    height = imag.shape[0]
    width = imag.shape[1]
    new_height = height*factor
    new_width = width*factor
    
    if imag.ndim == 3:  # Si es a color
        channels = imag.shape[2]
        zeros = np.zeros((new_height, new_width, channels), dtype=imag.dtype)
    else:  # Si es blanco y negro 
        zeros = np.zeros((new_height, new_width), dtype=imag.dtype)
         
    for i in range(new_height):
        for j in range(new_width):
            orig_i = i // factor
            orig_j = j // factor
            zeros[i, j] = imag[orig_i, orig_j]
            
    saveimage(zeros, dic_param['output'])



def doAMF(params):
    dic_param = parse_params(params)


    imag = readimage(dic_param['input'])

    m = int(dic_param['m'])
    n = int(dic_param['n'])

    j = imag.shape[0]
    k = imag.shape[1]


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
    # Manejamos si la imagen es a color (3D) o en escala de grises (2D)
    if imag.ndim == 3:
        for x in range(offset_m, height - offset_m):
            for y in range(offset_n, width - offset_n):
                # Ventana dinámica de m x n centrada en (x, y)
                window = imag[x - offset_m : x + offset_m + 1, y - offset_n : y + offset_n + 1, :]
                vmin = np.min(window, axis=(0, 1))
                vmax = np.max(window, axis=(0, 1))
                imag_new[x, y] = (vmax.astype(np.float32) + vmin.astype(np.float32)) / 2
    else:
        for x in range(offset_m, height - offset_m):
            for y in range(offset_n, width - offset_n):
                # Ventana dinámica de m x n centrada en (x, y)
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
        
        
        
    
        
    
    




if len(sys.argv) == 1:
    print("No command line parameters given.\n")
    sys.exit()

if len(sys.argv) == 2:
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
    

