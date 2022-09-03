import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
from random import shuffle
import os
import time
import argparse
from  constants import DEEPFASHION_KEYPOINTS, DEEPFASHION_SKELETON
import shutil

# this code uses the annotations from the attribute and category detection task
parser = argparse.ArgumentParser(description='DeppFashion 2 COCO :)')
parser.add_argument('--dataset-root', help='root of the deepfashion dataset', default='')
parser.add_argument('--root-save', help='root where to save the json', default='')
args = parser.parse_args()


class convert_deepfashion_to_coco:
    def __init__(self):
        self.json_file = {}
        
    def initiate_json(self):
        """
        Initiate json file: one for training phase and another one for validation.
        """
        self.json_file[u'info'] = dict(url="https://github.com/openpifpaf/openpifpaf",
                                  date_created=time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime()),
                                  description="Conversion of DeepFashion dataset into MS-COCO format")
        self.json_file[u'categories'] = [dict(name='',  # Category name
                                         id=1,  # Id of category
                                         skeleton=[],  # Skeleton connections (check constants.py)
                                         supercategory='',  # Same as category if no supercategory
                                         keypoints=[])]  # Keypoint names
                                         
        self.json_file[u'images'] = []  # Empty for initialization
        self.json_file[u'annotations'] = []  # Empty for initialization

    def to_coco(self, shuffle_idx, deepfashion_root, datatype, root_save):
        # start separate:
        dir_name = 'img_' + datatype
        # root_save = '/work/vita/pegah/venv/pifpaf_deepfahsion'
        img_dir_path = os.path.join(root_save, dir_name)
        os.mkdir(img_dir_path)
        # end separate
        self.initiate_json()

        landmark_annofile = deepfashion_root + 'Anno/list_landmarks.txt'
        category_bboxfile = deepfashion_root + 'Anno/list_bbox.txt'
        print('category path ', category_bboxfile)
        
        
        landmarks = open(landmark_annofile).readlines()[2:]
        bboxs = open(category_bboxfile).readlines()[2:]
        assert len(landmarks) == len(bboxs)
        print ('Num of Deepfashion Category Images: ', len(landmarks))

        # print('keypoints are ', list(full_kps))
        self.json_file[u'categories'] = [{  "name":'clothing',
                                            "id":1,
                                            "skeleton":list(DEEPFASHION_SKELETON),
                                            "supercategory":'full',
                                            "keypoints":list(DEEPFASHION_KEYPOINTS)}]
        

        count = 0 # a counter to go over the every image
        for idx in shuffle_idx:
            count += 1
            im_name = '%.6d.png' % idx
            output_path = deepfashion_root + 'whathappening.txt' # => the path seems to be right!!
            with open(output_path, 'a') as f:
                json.dump(im_name, f)
            # bboxs[idx].split(' ')[0] is the number of images
            assert bboxs[idx].split(' ')[0] == landmarks[idx].split(' ')[0]

            # finding the path of the image and its size:
            image_name = landmarks[idx].split(' ')[0] # the first part of the line idx is the addr
            image_file = deepfashion_root + image_name
            # print('this is the img file ', image_file)
            
            if os.path.isfile(image_file):
                # start separate
                error = 0
                class_dir_name = img_dir_path + '/' + im_name
                
                shutil_origin = image_file
                shutil_dest = class_dir_name

                
                try:
                    shutil.copy(shutil_origin, shutil_dest)
                except IOError as e:
                    error = 1

                    
                # end separate
                if error == 0:
                    img = cv2.imread(image_file, 0)
                    if (img == None).any():
                        with open(output_path, 'w') as f:
                            json.dump('it is none.....'+image_file, f)
                    # print(type(img))
                    # let us try with the following function
                    # img = cv2.cv.LoadImage(image_file, cv2.IMREAD_COLOR) # did not work
                    height, width = img.shape
                    # print('this is the height ', height, ' this is the width ', width)
                    x1, y1, x2, y2 = map(int, bboxs[idx].split(' ')[-4:])
                    # print('corners ', x1, ' ', y1, ' ', x2, ' ', y2)
                    area = (x2 - x1) * (y2 - y1)
                    bbox = [x1, y1, x2 - x1, y2 - y1]

                    ''' 
                        we don't care about variation type here
                        variation type = "1" represents normal pose, "2" represents medium pose, "3" represents large pose, "4" represents medium zoom-in, "5" represents large zoom-in
                        visibility =  "0" represents visible, "1" represents invisible/occluded, "2" represents truncated/cut-off'''
                    
                    after_addr = landmarks[idx].split()[1:]
                    cloth_type = int(after_addr[0])
                    
                    keypoints = []
                    # extracting the key points:
                    # we want to have the same number of keypoints for all the categories even thought they 
                    # may not have all those => consider those that are not present as 0 => occluded or not visible
                    



                    # unlike the older works, we assign the last set of keypoints in the upper body to be hem and not waistline
                    if cloth_type == 1: # it does not have hem
                        
                        for i in range(0, 6):
                            print(after_addr, '   and here')
                            current_visib = (int(after_addr[3*i+2])*2+2)%3
                            current_x = after_addr[3*i+3]
                            current_y = after_addr[3*i+4]
                            keypoints.append(int(current_x))
                            keypoints.append(int(current_y))
                            keypoints.append(int(current_visib))
                        for i in range(2):
                            current_visib = 0
                            current_x = 0
                            current_y = 0
                            keypoints.append(int(current_x))
                            keypoints.append(int(current_y))
                            keypoints.append(int(current_visib))
                       

                    if cloth_type == 2: # it does not have collar and sleeve
                        index_visib = 2
                        index_x = 3
                        index_y = 4
                        for i in range(4): # for the collar and sleeve
                            current_visib = 0
                            current_x = 0
                            current_y = 0
                            keypoints.append(int(current_x))
                            keypoints.append(int(current_y))
                            keypoints.append(int(current_visib))
                        for i in range(4): # only waistlines and hems
                            current_visib = (int(after_addr[index_visib])*2+2)%3
                            current_x = after_addr[index_x]
                            current_y = after_addr[index_y]
                            index_visib += 3
                            index_x += 3
                            index_y += 3
                            keypoints.append(int(current_x))
                            keypoints.append(int(current_y))
                            keypoints.append(int(current_visib))

                    if cloth_type == 3: # it does have all the kps
                        index_visib = 2
                        index_x = 3
                        index_y = 4
                        for i in range(8):
                            current_visib = (int(after_addr[index_visib])*2+2)%3
                            current_x = after_addr[index_x]
                            current_y = after_addr[index_y]
                            index_visib += 3
                            index_x += 3
                            index_y += 3
                            keypoints.append(int(current_x))
                            keypoints.append(int(current_y))
                            keypoints.append(int(current_visib))


            

                    # completing the annotations:
                    self.json_file[u'annotations'].append({
                        'image_id': int(im_name[:-4]), # it neglects the .png
                        'category_id': 1,
                        'iscrowd': 0,
                        'id': idx, # I am not sure about this part
                        'area': area,
                        'bbox': list(bbox),
                        'num_keypoints': 8,
                        'keypoints': keypoints,
                        'segmentation': [[312.29, 562.89]]})

                    '''{"image_id": 245419, "category_id": "clothing", "iscrowd": 0,
                    "id": 245419, "area": 39424, "bbox": [55, 44, 154, 256],
                    "num_keypoints": 8, "keypoints": [101, 71, 2, 133, 66, 2, 82, 82, 2, 154, 70, 2, 91, 158, 2, 158, 155, 2, 76, 283, 2, 183, 283, 2],
                    "segmentation": [[312.29, 562.89]]}'''

                    # completing the images
                    self.json_file[u'images'].append({
                        'coco_url': "unknown",
                        'file_name': im_name,
                        'id': int(im_name[:-4]),
                        'license': 1,
                        'date_captured': "unknown",
                        'width': width,
                        'height': height}
                        )
        new_file = root_save + '/' + datatype + "_annotations_MSCOCO_style.json"
        with open(new_file, 'w') as f:
            json.dump(self.json_file, f)


# /Users/pegahkhayatan/Desktop/datasets/openpifpaf_deepfashion/to_coco_and_separate.py
if __name__ == '__main__':
    print ('Welcome to DeepFashion2COCO :))')

    # deepfashion_root = 'to be precised', given as argument with default
    deepfashion_root = args.dataset_root
    root_save = args.root_save
    '''
    num_valid = 89000
    num_train = 289222 - 89000
    '''
    img_partition = deepfashion_root + 'Eval/list_eval_partition.txt'
    partitions = open(img_partition).readlines()[2:]
    index_train = []
    index_eval = []
    index_test = []
    for i, partition in enumerate(partitions):
        #print(partition)
        cat = partition.split()[1]
        if cat == 'train':
            index_train.append(i)

        elif cat == 'val':
            index_eval.append(i)

        elif cat == 'test':
            index_test.append(i)

   
    deep2coco = convert_deepfashion_to_coco() 
    deep2coco.initiate_json()
    coco_dict_train = deep2coco.to_coco(index_train, deepfashion_root, 'train', root_save)
    coco_dict_val = deep2coco.to_coco(index_eval, deepfashion_root, 'valid', root_save)
    coco_dict_test = deep2coco.to_coco(index_test, deepfashion_root, 'test', root_save)
    #shuffle(shuffle_idx)

    
    print('We did what had to be done!!')