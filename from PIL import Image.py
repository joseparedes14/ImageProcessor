from PIL import Image

def hflip(img):
    width, height = img.size
    pixels = img.load()
    new = Image.new(img.mode, (width, height))
    pixels_new = new.load()
    for y in range(height):
        for x in range(width):
            pixels_new[width - 1 - x, y] = pixels[x, y] 
    return new

# Asegúrate de cambiar 'input.jpg' por el nombre real de tu imagen en esa carpeta
try:
    img = Image.open("input.jpg")
    resultado = hflip(img)
    resultado.save("output.jpg")
    print("¡Imagen volteada con éxito! Guardada como output.jpg")
except FileNotFoundError:
    print("No se encontró la imagen 'input.jpg'. Cambia el nombre en el script por una imagen existente.")