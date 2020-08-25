import sys
import os
import easyocr

reader = easyocr.Reader(['ch_tra'])

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
print(out)
print(detect_count/img_count)