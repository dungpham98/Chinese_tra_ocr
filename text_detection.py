import sys
from collections import OrderedDict
import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import imgproc
from craft import CRAFT
import craft_utils
import os

import easyocr

reader = easyocr.Reader(['ch_tra'])

img_count = 0
detect_count = 0
out = []


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
        list_images = []
        
        
        boxs = boxes
        
        for i in range(len(boxs)):
            boxs[i][0][0] = boxes[i][0][0]-20
            boxs[i][3][0] = boxes[i][3][0]-20
            boxs[i][1][0] = boxes[i][1][0]+20
            boxs[i][2][0] = boxes[i][2][0]+20

            boxs[i][0][1] = boxes[i][0][1]-20
            boxs[i][3][1] = boxes[i][3][1]-20
            boxs[i][1][1] = boxes[i][1][1]+20
            boxs[i][2][1] = boxes[i][2][1]+20
            print(boxs[i])
        
        b_x = []
        b_y = []

        m_xmax = 0
        m_ymax = 0
        m_xmin = 100000
        m_ymin = 100000

        
        for i, _b in enumerate(boxes):
            b = np.array(_b, dtype=np.int16)
            xmin = np.min(b[:, 0])
            ymin = np.min(b[:, 1])

            xmax = np.max(b[:, 0])
            ymax = np.max(b[:, 1])
            x_m = xmax-xmin
            y_m = ymax-ymin
            if(m_xmax < xmax):
                m_xmax = xmax
            if(m_ymax < ymax):
                m_ymax = ymax

            if(m_xmin > xmin):
                m_xmin = xmin
            if(m_ymin > ymin):
                m_ymin = ymin

            b_x.append(x_m)
            b_y.append(y_m)

            r = image[ymin:ymax, xmin:xmax, :].copy()
            list_images.append(r)

        b_x = np.array(b_x)
        b_y = np.array(b_y)

        ave_x = np.mean(b_x)
        ave_y = np.mean(b_y)

        print(m_xmax)
        print(ave_x)
        m_x = int(m_xmax/ave_x)
        print(m_ymax)
        print(ave_y)
        m_y = int(m_ymax/ave_y)

        m = [[" " for x in range(m_x)] for y in range(m_y)]
        m = np.array(m)
        print(m.shape)
        for i, _b in enumerate(boxes):
            b = np.array(_b, dtype=np.int16)
            xmin = np.min(b[:, 0])
            ymin = np.min(b[:, 1])

            xmax = np.max(b[:, 0])
            ymax = np.max(b[:, 1])

            x_i = int(xmin/ave_x)
            y_i = int(ymin/ave_y)

            print(str(x_i)+"  "+str(y_i))

            m[y_i][x_i] = "X"

        print(m.shape)

        a = m
        mat = np.matrix(a)
        with open('outfile.txt','wb') as f:
            for line in mat:
                np.savetxt(f, line, fmt='%s')
    
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

        return boxes, list_images, m


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
        bboxes, list_crop_images, text_matrix = detection.get_bounding_box('%s/%s' % (input_img_folder, fname), False)

     
        # Sort output as increase of bboxes
        sort_array = [(bboxes[i][0][0], bboxes[i][0][1], i) for i in range(len(bboxes))]
        sort_array = sorted(sort_array)

        basename = os.path.basename(fname)[:-4]
        for i in range(len(sort_array)):
            j = sort_array[i][2]
            image = list_crop_images[j]

            output_file = "%s/%s_%i.jpg" % (output_folder, basename, i)
            cv2.imwrite(output_file, image)

        for image in os.listdir(sys.argv[2]):
            image = sys.argv[2]+image
            img_count = img_count + 1
            output = reader.readtext(image)
            if(output != []):
                if(output[0][1] != ''):
                    detect_count = detect_count + 1
                    out.append(output[0][1])
        print(out)
        print(detect_count/img_count)



if __name__ == "__main__":

    # detection = TextDetection()
    # detection.get_bounding_box(sys.argv[1], True)

    process_image_folders(sys.argv[1], sys.argv[2])
