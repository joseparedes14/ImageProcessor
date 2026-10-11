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

# G1
def hflip(img):
    # we first get the image dimmensions
    width, height = img.size
    # we get the pixels
    pixels = img.load()
    # we create a new image to save the result of the gorizontal flip
    new = Image.new(img.mode, (width, height))
    # -> im.mode to conserve the type of pixels of the original image, for example RGB
    # and we get the access to the new image's pixels
    pixels_new =  new.load()
    # change positions
    for y in range(height):
        for x in range(width):
            pixels_new[width - 1 - x, y] = pixels[x, y] 
    return new

# G2
def vflip(img):
    # we first get the image dimmensions
    width, height = img.size
    # we get the pixels
    pixels = img.load()
    # we create a new image to save the result of the gorizontal flip
    new = Image.new(img.mode, (width, height))
    # -> im.mode to conserve the type of pixels of the original image, for example RGB
    # and we get the access to the new image's pixels
    pixels_new =  new.load()
    # change positions
    for y in range(height):
        for x in range(width):
            pixels_new[x, height-1-y] = pixels[x, y] 
    return new


# G3
def dflip(img):
    # we first get the image dimmensions
    width, height = img.size
    # we get the pixels
    pixels = img.load()
    # we create a new image to save the result of the gorizontal flip
    new = Image.new(img.mode, (width, height))
    # -> im.mode to conserve the type of pixels of the original image, for example RGB
    # and we get the access to the new image's pixels
    pixels_new =  new.load()
    # change positions
    for y in range(height):
        for x in range(width):
            pixels_new[y, x] = pixels[x, y] 
    return new



# G4
def shrink(img):
    # we first get the image dimmensions
    width, height = img.size
    new_width = width//2
    new_height = height//2
    # we get the pixels
    pixels = img.load()
    # we create a new image to save the result
    new = Image.new(img.mode, (new_width, new_height))
    pixels_new = new.load()
    for y in range(new_height):
        for x in range(new_width):
            pixels_new[x, y] = pixels[2*x, 2*y]
    return new


#G5
def enlarge(img):
    # we first get the image dimmensions
    width, height = img.size
    new_width = width*2
    new_height = height*2
    # we get the pixels
    pixels = img.load()
    # we create a new image to save the result
    new = Image.new(img.mode, (new_width, new_height))
    pixels_new = new.load()
    for y in range(new_height):
        for x in range(new_width):
            pixels_new[x, y] = pixels[x//2, y//2]
    return new


# N1

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



