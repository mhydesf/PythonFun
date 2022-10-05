from PIL import Image, ImageFont, ImageDraw
import cv2
import numpy as np

DENSITY = 'Ñ@#W$9876543210?!abc;:+=-,._ '

def map_val(val: int,
        from_low: int,
        from_high: int,
        to_low: int,
        to_high: int) -> int:
    ratio = val / (from_high - from_low)
    return int(ratio * (to_high - to_low))

def resize_img(img: np.ndarray,
               ratio: int) -> np.ndarray:
    w, h = img.shape
    new_w = int(w/ratio)
    new_h = int(h/ratio)
    return cv2.resize(img, (new_w, new_h))


img = cv2.imread('guitar.JPG', cv2.IMREAD_GRAYSCALE)
img = resize_img(img=img, ratio=10)

myimg = Image.new('L', img.shape, color='black')
image_edit = ImageDraw.Draw(myimg)
font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/Ubuntu-L.ttf", 1)

for i, row in enumerate(img):
    for j, val in enumerate(row):
        image_edit.text(xy=(i, j),
                        text=DENSITY[map_val(val, 0, 255, len(DENSITY), 0) + 1],
                        fill=255,
                        font=font)
        
myimg.save('test.png')
