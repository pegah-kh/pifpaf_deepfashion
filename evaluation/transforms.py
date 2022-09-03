import copy
import logging
from turtle import right
import warnings
from openpifpaf.annotation import Annotation
import numpy as np
import openpifpaf
import PIL


LOG = logging.getLogger(__name__)


class BBoxCrop(openpifpaf.transforms.Preprocess):

    def __call__(self, image, anns, meta):
        # croping the bbox, first preprocessing
        # so the image should be like x and y and bbox is like the same
        print('cropping the bbox, first step of preprocessing')

        landmarks = anns[0]['keypoints']
        x_1 = anns[0]['bbox'][0]
        y_1 = anns[0]['bbox'][1]

        top = y_1
        left = x_1
        new_h = anns[0]['bbox'][3]
        new_w = anns[0]['bbox'][2]
        meta['original_left'] = left
        meta['original_top'] = top
        print('left, top, new_w, new_h ', left, '  ', top, '  ', new_w, '  ', new_h)

        im_np = np.asarray(image)
        print('debug 1 ', im_np.shape)
        meta['original_width'] = im_np.shape[1]
        meta['original_height'] = im_np.shape[0]
        im_np = im_np[top: top + new_h, left: left + new_w]
        meta['width_proportion'] = np.float(new_w/meta['original_width'])
        meta['height_proportion'] = np.float(new_h/meta['original_height'])
        image = PIL.Image.fromarray(im_np)
        
        
        temp_sides = [left, top]

        def is_coordinate(index):
            if (index+1)%3 > 0:
                return 1
            else:
                return 0
        landmarks = [(landmarks[3*i+j] - temp_sides[j%2]*(is_coordinate(j))) for i in range(8) for j in range(3)]
        anns[0]['keypoints'] = landmarks
        
        return image, anns, meta

class CheckLandmarks(openpifpaf.transforms.Preprocess):

    def __call__(self, image, anns, meta):
        w, h = image.size
        landmarks = (anns[0]['keypoints']).copy()
        
        for i in range(8):
            if (landmarks[3*i] < 0) or (landmarks[3*i] >= w) or (landmarks[3*i+1] < 0) or (landmarks[3*i+1] >= h):
                landmarks[3*i] = 0
                landmarks[3*i+1] = 0
                # but it did not matter if they were even negative, only the visibility should be set to zero
                landmarks[3*i+2] = 0
        
        anns[0]['keypoints'] = landmarks
        return image, anns, meta