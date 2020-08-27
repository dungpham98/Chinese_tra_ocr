import sys
from collections import OrderedDict
import cv2
import numpy as np
import json
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import imgproc
from craft import CRAFT
import craft_utils
import os

import easyocr

from typing import Tuple
import PIL  
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

reader = easyocr.Reader(['ch_tra'])

img_count = 0
detect_count = 0
out = []

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
                     text_rgb_color=(255, 0, 0), text_size=24, font='../chinese-ocr/simsun.ttc', **option):
    """
    USAGE:
        cv2_img_add_text(img, '中文', (0, 0), text_rgb_color=(0, 255, 0), text_size=12, font='mingliu.ttc')
    """
    pil_img = img
    if isinstance(pil_img, np.ndarray):
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_text = ImageFont.truetype('../chinese-ocr/simsun.ttc', size=text_size, encoding=option.get('encoding', 'utf-8'))
    draw.text(left_corner, text, text_rgb_color, font=font_text)
    cv2_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    if option.get('replace'):
        img[:] = cv2_img[:]
        return None
    return cv2_img

def copy_state_dict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


class TextDetection:

    def __init__(self):
        
        self.trained_model = '../chinese-ocr/weights/craft_mlt_25k.pth'
        self.text_threshold = 0.75
        self.low_text = 0.6
        self.link_threshold = 0.9
        self.cuda = True
        self.canvas_size = 1280
        self.mag_ratio = 1.5
        self.poly = False
        self.show_time = False

        self.net = CRAFT()

        self.net.load_state_dict(copy_state_dict(torch.load(self.trained_model)))
        self.net = self.net.cuda()

        cudnn.benchmark = False

        self.net.eval()

    def get_bounding_box(self, image_file, verbose=False):
        """
        Get the bounding boxes from image_file
        :param image_file
        :param verbose
        :return:
        """
        image = cv2.imread(image_file)
        img_dim = image.shape
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image,
                                                                              self.canvas_size,
                                                                              interpolation=cv2.INTER_LINEAR,
                                                                              mag_ratio=self.mag_ratio)

        ratio_h = ratio_w = 1 / target_ratio
        
        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        if self.cuda:
            x = x.cuda()

        # forward pass
        with torch.no_grad():
            y, feature = self.net(x)

        # make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link,
                                               self.text_threshold,
                                               self.link_threshold,
                                               self.low_text,
                                               self.poly)

        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        print(boxes)

        center_point = []
        for i, _b in enumerate(boxes):
            b = np.array(_b, dtype=np.int16)
            xmin = np.min(b[:, 0])
            ymin = np.min(b[:, 1])

            xmax = np.max(b[:, 0])
            ymax = np.max(b[:, 1])
            x_m = xmin+(xmax-xmin)/2
            y_m = ymin+(ymax-ymin)/2
            center_point.append([x_m,y_m])

        list_images = get_box_img(boxes,image)

        if verbose:
            for _b in boxes:
                b = np.array(_b, dtype=np.int16)
                xmin = np.min(b[:, 0])
                ymin = np.min(b[:, 1])

                xmax = np.max(b[:, 0])
                ymax = np.max(b[:, 1])

                r = image[ymin:ymax, xmin:xmax, :].copy()
                cv2.imshow("Crop", r)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    return

                cv2.destroyWindow("Crop")
        print(boxes)
        return boxes,list_images, center_point, img_dim

def get_box_img(boxes,image):
    #print(center_point)
    list_images = []
    boxs = np.copy(boxes)
    
    for i in range(len(boxs)):
        boxs[i][0][0] = boxes[i][0][0]-20
        boxs[i][3][0] = boxes[i][3][0]-20
        boxs[i][1][0] = boxes[i][1][0]+20
        boxs[i][2][0] = boxes[i][2][0]+20

        boxs[i][0][1] = boxes[i][0][1]-20
        boxs[i][3][1] = boxes[i][3][1]-20
        boxs[i][1][1] = boxes[i][1][1]+20
        boxs[i][2][1] = boxes[i][2][1]+20
    
    for i, _b in enumerate(boxs):
        b = np.array(_b, dtype=np.int16)
        xmin = np.min(b[:, 0])
        ymin = np.min(b[:, 1])

        xmax = np.max(b[:, 0])
        ymax = np.max(b[:, 1])

        r = image[ymin:ymax, xmin:xmax, :].copy()
        list_images.append(r)

    return list_images

def detect_symbol(filename,np_img,cen_point,idx):
    draw_text = init_parameters(cv2_img_add_text, text_size=100, text_rgb_color=(0, 0, 255), font='kaiu.ttf', replace=True)
    detect = False
    output = reader.readtext(filename)
    if(output != []):
        if(output[0][1] != ''):
            detect = output[0][1]
            draw_text(np_img, output[0][1], (cen_point[idx][0], cen_point[idx][1]))

    return detect

def draw_lines(np_img,sort_array):
    pivot = sort_array[0][0]
    start = (sort_array[0][0],sort_array[0][1])
    max_y = 0
    min_y = 100000
    for i in range(len(sort_array)-12):
        if(max_y < sort_array[i][1]):
            max_y = sort_array[i][1]
        if(min_y > sort_array[i][1]):
            min_y = sort_array[i][1]
    max_y = int(max_y + 100)
    for i in range(len(sort_array)-1):
        diff = sort_array[i+1][0]-sort_array[i][0]
        if(diff > 50 and diff < 200):
            end = (sort_array[i][0],max_y)
            image = cv2.line(np_img, start, end, (0,0,0), 5)
            start = (sort_array[i+1][0],sort_array[i+1][1])
            
    cv2.imshow("sample", image)  
    return image

def add_JSON_detect(d,box):
    region = {
        "shape_attributes": {
          "name": "polygon",
          "all_points_x": [
            float(box[0][0]),
            float(box[1][0]),
            float(box[2][0]),
            float(box[3][0])
          ],
          "all_points_y": [
            float(box[0][1]),
            float(box[1][1]),
            float(box[2][1]),
            float(box[3][1])
          ]
        },
        "region_attributes": {
          "name": d
        }
      }
    return region

def to_JSON(regions,img_name,size):

    img_name = "/home/tuandung/chinese-ocr/data/test_img/"+img_name

    res = {
        img_name: {
            "filename": img_name,
            "size": size,
            "regions": regions,
            "file_attributes": {}
          }
        }

    return res


def process_image_folders(input_img_folder, output_folder):
    """
    Give an input image folder, process each image and output to output folder
    Args:
        input_img_folder:
        output_folder:
    """

    # Check whether exists the input image
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    print("Init detector")
    detection = TextDetection()

    list_files = os.listdir(input_img_folder)
    print("Got %i files" % len(list_files))

    for fname in list_files:
        print("Process ", fname)
        bboxes, list_crop_images, cen_point, img_dim = detection.get_bounding_box('%s/%s' % (input_img_folder, fname), False)
        np_img = np.ones(img_dim, dtype=np.uint8) * 255  # background with white color
        img_count = 0
        detect_count = 0
        file_size = Path('%s/%s' % (input_img_folder, fname)).stat().st_size
        # Sort output as increase of bboxes
        sort_array = [(bboxes[i][0][0], bboxes[i][0][1], i) for i in range(len(bboxes))]
        sort_array = sorted(sort_array)

        ## down and right
        basename = os.path.basename(fname)[:-4]
        regions = []
        for i in range(len(sort_array)):
            j = sort_array[i][2]
            image = list_crop_images[j]

            output_file = "%s/%s_%i.jpg" % (output_folder, basename, i)
            cv2.imwrite(output_file, image)

            d = detect_symbol(output_file,np_img,cen_point,j)
            if(d):
                region = add_JSON_detect(d,bboxes[j])
                regions.append(region)
                detect_count = detect_count+1

        res = to_JSON(regions,fname,file_size)
        json_detect = json.dumps(res, indent=4, sort_keys=True)
        with open('boxes.json', 'w') as outfile:
            json.dump(res, outfile)
        draw_lines(np_img,sort_array)

        print("detection rate: "+ str(detect_count/len(sort_array)))
        cv2.imwrite("out_det.jpg",np_img)

      

if __name__ == "__main__":

    # detection = TextDetection()
    # detection.get_bounding_box(sys.argv[1], True)

    process_image_folders(sys.argv[1], sys.argv[2])