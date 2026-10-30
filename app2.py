# ----- Libraries -----

from PIL import Image
import sys
import numpy as np
from datetime import datetime


# ------ 


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
        default_name = f"images_output/result-{timestamp}.bmp"
        newIm.save(default_name)


def parse_params(params):
    dic_param = dict()
    for param in params:
        values = param.strip('-').split('=')
        dic_param[values[0]] = values[1]
    return dic_param





# H1


# C1


# C2


# C3



# C4


# C5


# C6



# S4



# O3




















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
    pass
