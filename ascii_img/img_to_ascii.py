import os
import argparse
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
    return cv2.resize(img, (new_h, new_w))

def crop_img(img: np.ndarray,
             start_x: int=None,
             end_x: int=None,
             start_y: int=None,
             end_y: int=None) -> np.ndarray:
    x, y = img.shape
    if start_x is None:
        start_x = 0
    if end_x is None:
        end_x = x
    if start_y is None:
        start_y = 0
    if end_y is None:
        end_y = y
        
    return img[start_x:end_x, start_y:end_y]

if __name__ == "__main__":
    
    DESCRIPTION="Conversion of a pixel image into an image of ASCII characters"
    
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('input')
    parser.add_argument('-o', '--output', default=None)
    parser.add_argument('-r', '--ratio', default=1)
    args = parser.parse_args()
    
    file = str(args.input)
    if args.output is None:
        output = f"ascii_{file.split('.')[0]}.txt"
    else:
        output = str(args.output)
    ratio = int(args.ratio)

    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = resize_img(img=img, ratio=ratio)
    #img = crop_img(img, 0, img.shape[0]-50, img.shape[1] - 275, img.shape[1]-25)
    print(img.shape)

    cv2.imshow('test', img)
    cv2.waitKey(0)

    ascii_img = []

    for row in img:
        row_str = ''
        for val in row:
            row_str += DENSITY[map_val(val, 0, 255, len(DENSITY), 0) + 1]
        ascii_img.append(row_str)

    if os.path.exists(output):
        os.remove(output)

    with open(output, 'w+') as f:
        for row in ascii_img:
            f.write(row+'\n')