# ----- Libraries -----

from PIL import Image
import sys
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
# ------ 


L = 256


def readimage(path):
    im = Image.open(path)
    
    
    arr = np.array(im.getdata())
   
    if arr.ndim == 1: 
       
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



# Building the histogram and saving it


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





# H1

def doHUniform(params):
    histogram = doHistogram(params)
    dic_param = parse_params(params)
    img = readimage(dic_param['input'])
    s = np.zeros_like(histogram)

    for i in range(len(s)):
        s[i] = (L-1)*np.sum(histogram[0:i+1])
        s = np.round(s).astype(np.uint8)

    img_enhanced = s[img]

    
    output_path = dic_param.get('output', 'images_output/enhanced_uniform.png')
    saveimage(img_enhanced, output_path) # Asegúrate de usar tu función para guardar imágenes

    return img_enhanced











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






if command == '--histogram':
    doHistogram(param)

elif command == '--huniform':
    doHUniform(param)