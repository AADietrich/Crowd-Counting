import random
import PIL.Image as pil
import params as p

def crop(image_path):
    image = pil.open(image_path)
    w,h = image.size
    x,y = p.PATCH_X, p.PATCH_Y
    if(x>=w or y>=h):
        return
    left = random.randint(0,w-x-1)
    top = random.randint(0,h-y-1)
    cropped = image.crop((left, top, left+x, top+y))
    return cropped
