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
    if arr.ndim == 1: #grayscale
        numColorChannels = 1
        arr = arr.reshape(im.size[1], im.size[0])
    else:
        numColorChannels = arr.shape[1]
        arr = arr.reshape(im.size[1], im.size[0], numColorChannels)

    return arr


def saveimage(arr, output=None):

    newIm = Image.fromarray(arr.astype(np.uint8))

    if output:
        newIm.save(output)
    else:

        newIm.save(f"result-{datetime.now()}.bmp")




def doBrightness(params):
    dic_param = dict()
    for param in params:
        values = param.strip('-').split('=')
        dic_param[values[0]] = values[1]

    imag = readimage(dic_param['input'])

    constant = int(dic_param['value'])

    if constant <= 0:
        imag = np.maximum(0, imag + constant)
    else:
        imag = np.minimum(255, imag + constant)

    saveimage(imag, dic_param['output'])



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

