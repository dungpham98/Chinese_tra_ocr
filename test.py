import sys
import os
import easyocr

from typing import Tuple
from PIL import Image
from PIL import ImageDraw
import PIL  
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


# define decorator
def init_parameters(fun, **init_dict):
    """
    help you to set the parameters in one's habits
    """
    def job(*args, **option):
        option.update(init_dict)
        return fun(*args, **option)
    return job


def cv2_img_add_text(img, text, left_corner: Tuple[int, int],
                     text_rgb_color=(255, 0, 0), text_size=24, font='/home/tuandung/chinese-ocr/simsun.ttc', **option):
    """
    USAGE:
        cv2_img_add_text(img, '中文', (0, 0), text_rgb_color=(0, 255, 0), text_size=12, font='mingliu.ttc')
    """
    pil_img = img
    if isinstance(pil_img, np.ndarray):
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_text = ImageFont.truetype('/home/tuandung/chinese-ocr/simsun.ttc', size=text_size, encoding=option.get('encoding', 'utf-8'))
    draw.text(left_corner, text, text_rgb_color, font=font_text)
    cv2_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    if option.get('replace'):
        img[:] = cv2_img[:]
        return None
    return cv2_img


reader = easyocr.Reader(['ch_tra'])

"""
img_count = 0
detect_count = 0
out = []
for image in os.listdir(sys.argv[1]):
	image = sys.argv[1]+image
	img_count = img_count + 1
	output = reader.readtext(image)
	if(output != []):
		if(output[0][1] != ''):
			detect_count = detect_count + 1
			out.append(output[0][1])
			img = Image.new('RGB', (200, 100))
			d = ImageDraw.Draw(img)
			d.text((20, 20), output[0][1], fill=(255, 0, 0))
			img = img.save("test.jpg")
			break
print(out)
print(detect_count/img_count)
"""
img_count = 0
detect_count = 0
out = "a"
for image in os.listdir(sys.argv[1]):
	image = sys.argv[1]+image
	img_count = img_count + 1
	output = reader.readtext(image)
	if(output != []):
		if(output[0][1] != ''):
			detect_count = detect_count + 1
			out = output[0][1]
			break

np_img = np.ones((64, 32, 3), dtype=np.uint8) * 255  # background with white color
draw_text = init_parameters(cv2_img_add_text, text_size=32, text_rgb_color=(0, 0, 255), font='kaiu.ttf', replace=True)
draw_text(np_img, out, (0, 0))
cv2.imshow('demo', np_img), cv2.waitKey(0)