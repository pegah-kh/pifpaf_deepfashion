from __future__ import annotations
from email.mime import image
import os
import glob
import json
import copy
import logging
import csv
import time
from collections import defaultdict

import numpy as np
import torch
import PIL
from PIL import Image
from openpifpaf import decoder, network, visualizer, show, logger, Predictor

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from contextlib import contextmanager
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from scipy import ndimage
except ImportError:
    ndimage = None

DEEPFASHION_SKELETON = [
    [1, 2], # left and right collar connection
    [1, 3],
    [3, 5], # left hem and left sleeve connection
    [5, 6], # left and right hem connection I AM NOT SURE IF THIS SHOULD BE IN THE SKELETON
    [6, 4], # right sleeve and right collar connection
    [4, 2],
    [5, 7], 
    [7, 8],
    [8, 6],
]



import cv2

try:
    import gdown
    DOWNLOAD = copy.copy(gdown.download)
except ImportError:
    DOWNLOAD = None


LOG = logging.getLogger(__name__)


def factory_from_args(args):

    logger.configure(args, LOG)  

    # Devices
    args.device = torch.device('cpu')
    args.pin_memory = False
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    LOG.debug('neural network device: %s', args.device)

    # Add visualization defaults    
    args.figure_width = 10
    args.dpi_factor = 1.0

    args.loader_workers = None

    # Configure
    decoder.configure(args)
    network.Factory.configure(args)
    Predictor.configure(args) 
    show.configure(args)
    visualizer.configure(args)

    return args


def predict(args, visualize = False):
   
    args = factory_from_args(args)
    predictor = Predictor(checkpoint=args.checkpoint)

    annotations_accumul = []
    total_output = []
    sample_landmarks = args.landmarks
    sample_visibs = args.vis


    batch_size = 0
    for batch_n, (pred, _, _) in enumerate(predictor.numpy_images(args.images)):
        width, height = args.images[batch_n].shape[1], args.images[batch_n].shape[2]
        width = width/1.0
        height = height/1.0
        batch_size += 1
        annotations_accumul = [ann.json_data() for ann in pred]
        best_local_score = 0
        best_local_annotation = []
        annotation_found = False

            
        for dict in annotations_accumul:
            if dict['score'] > best_local_score:
                annotation_found = True
                best_local_score = dict['score']
                best_local_annotation = dict['keypoints']

        if annotation_found:
            total_output.append(best_local_annotation)
        else:
            #           x,      y,      v
            # collar_l  w/2     h/4     2
            # collar_r  w/2     h/4     2
            # sleeve_l  w/4     h/2     2
            # sleeve_r  3w/4    h/2     2
            # hem_l     w/2     3h/4    2
            # hem_r     w/2     3h/4    2

            approximation = np.array([width/2, height/4, 2,
                                 width/2, height/4, 2,
                                 width/4, height/2, 2,
                                 3*width/4, height/2, 2,
                                 width/2, 3*height/4, 2,
                                 width/2, 3*height/4, 2,
                                 width/2, 3*height/4, 2,
                                 width/2, 3*height/4, 2]).astype(np.float64)
            total_output.append(approximation)
            # total_output.append([0]*24)



    converted_keypoints = [total_output[batch_n][3*k + j] for batch_n in range(batch_size) for k in range(8) for j in np.array([0,1])]
    converted_keypoints = np.array(converted_keypoints)
    converted_keypoints = np.reshape(converted_keypoints, (-1, 8, 2))
    converted_visibs = [total_output[batch_n][3*k + 2] for batch_n in range(batch_size) for k in range(8)]
    converted_visibs = np.array(converted_visibs)
    converted_visibs = np.reshape(converted_visibs, (-1, 8, 1))
    converted_visibs = np.where(converted_visibs > args.confidence_thresh, 1, 0)

    
    if visualize:
        in_pic_sample_vis = np.where(sample_visibs[0] >= 1, 1, 0)
        vis_gt =  np.reshape([in_pic_sample_vis], (-1, 8, 1))
        gt_lms = np.reshape([sample_landmarks[0]], (-1, 8, 2))

        img_converted_keypoints = [total_output[0][3*k + j] for k in range(8) for j in np.array([0,1])]
        img_converted_keypoints = np.array(img_converted_keypoints)
        img_converted_keypoints = np.reshape(img_converted_keypoints, (-1, 8, 2))


        annotated_image_pred, annotated_image_gt = kp_visualizer(args.temp_image_name, args.images[0], vis_gt, vis_gt, img_converted_keypoints, gt_lms)
        
        return converted_keypoints, converted_visibs, annotated_image_pred, annotated_image_gt
    
    return converted_keypoints, converted_visibs

def pifpaf_output(args, visualize = False): # lm_batch should be of size: batch_size * 8
    if visualize:
        converted_keypoints, converted_visibs, annotated_image_pred, annotated_image_gt = predict(args, visualize = True)
        return converted_keypoints, converted_visibs, annotated_image_pred, annotated_image_gt
    else:
        converted_keypoints, converted_visibs = predict(args)
        return converted_keypoints, converted_visibs
    # torch_output_lm_map = keypoint_transformer(keypoints_sets, args)


class LandmarkEvaluator(object):

    def __init__(self):

        self.reset()

    def reset(self):
     
        self.lm_vis_count_all = np.zeros((8, 1))
        self.lm_pic_count_all = np.zeros((8, 1))
        self.lm_vis_count_intersection = np.zeros((8, 1))
        self.lm_pic_count_intersection = np.zeros((8, 1))
        self.lm_vis_count_intersection_lowconf = np.zeros((8, 1))
        self.lm_pic_count_intersection_lowconf = np.zeros((8, 1))


        self.lm_vis_dist = np.zeros((8, 1))
        self.lm_pic_dist = np.zeros((8, 1))
        self.lm_vis_dist_intersection = np.zeros((8, 1))
        self.lm_pic_dist_intersection = np.zeros((8, 1))

        self.in_pred_not_gt = np.zeros((8, 1)) # you should change it to the right definition
        self.in_gt_not_pred = np.zeros((8, 1))
        self.in_pic_not_pred = np.zeros((8, 1))


        self.n_instances = 0


    def landmark_count(self, output, conf_preds, args):


        # evaluation_visib should be the multiplication of the two
        batch_size = output.shape[0]
        self.n_instances += batch_size


        mask_vis = np.where(args.vis==2, 1, 0)
        mask_pic = np.where(args.vis>=1, 1, 0)
        batch_lm_vis_count_all = np.reshape(mask_vis, (batch_size, 8, 1)) # (20, 8, 1)
        batch_lm_pic_count_all = np.reshape(mask_pic, (batch_size, 8, 1)) # (20, 8, 1)
        batch_lm_vis_count_intersection = np.multiply(batch_lm_vis_count_all, conf_preds) # (20, 8, 1)
        batch_lm_pic_count_intersection = np.multiply(batch_lm_pic_count_all, conf_preds) # (20, 8, 1)
        
        self.lm_vis_count_all += np.sum(batch_lm_vis_count_all, axis=0)
        self.lm_pic_count_all += np.sum(batch_lm_pic_count_all, axis=0)
        self.lm_vis_count_intersection += np.sum(batch_lm_vis_count_intersection, axis=0)
        self.lm_pic_count_intersection += np.sum(batch_lm_pic_count_intersection, axis=0)
        

        # in_gt_not_pred_local and in_pred_not_gt_local
        in_gt_not_pred_local = batch_lm_vis_count_all - conf_preds
        in_gt_not_pred_local = np.where(in_gt_not_pred_local==1, in_gt_not_pred_local, 0)
        in_gt_not_pred_local = np.sum(in_gt_not_pred_local, axis = 0)
        self.in_gt_not_pred += in_gt_not_pred_local

        in_pred_not_gt_local = conf_preds - batch_lm_vis_count_all
        in_pred_not_gt_local = np.where(in_pred_not_gt_local==1, in_pred_not_gt_local, 0)
        in_pred_not_gt_local = np.multiply(batch_lm_pic_count_all, in_pred_not_gt_local)
        in_pred_not_gt_local = np.sum(in_pred_not_gt_local, axis = 0)
        self.in_pred_not_gt += in_pred_not_gt_local


        in_pic_not_pred = batch_lm_pic_count_all - conf_preds
        in_pic_not_pred = np.where(in_pic_not_pred==1, in_pic_not_pred, 0)
        in_pic_not_pred = np.sum(in_pic_not_pred, axis = 0)
        self.in_pic_not_pred += in_pic_not_pred

        output = output.astype(np.float64)
        non_normalized_output = copy.copy(output)
        sample_landmarks = args.landmarks
        non_normalized_sample_landmarks = copy.copy(sample_landmarks)

        # we should add left tops
        '''for batch_n in range(batch_size):
            meta = args.metas[batch_n]
            left_top = np.array([meta['original_left'], meta['original_top']])
            left_top = np.array([left_top]*8).reshape((8, 2))
            non_normalized_output[batch_n] += left_top
            non_normalized_sample_landmarks[batch_n] += left_top'''


        '''print('debug 1 ', non_normalized_output, '    ', non_normalized_sample_landmarks)
        for batch_n in range(batch_size):
            meta = args.metas[batch_n]
            # offset
            non_normalized_output[batch_n][:, 0] += meta['offset'][0]
            non_normalized_output[batch_n][:, 1] += meta['offset'][1]

            non_normalized_sample_landmarks[batch_n][:, 0] += meta['offset'][0]
            non_normalized_sample_landmarks[batch_n][:, 1] += meta['offset'][1]
            

            # scale
            non_normalized_output[batch_n][:, 0] = non_normalized_output[batch_n][:, 0] / meta['scale'][0]
            non_normalized_output[batch_n][:, 1] = non_normalized_output[batch_n][:, 1] / meta['scale'][1]

            non_normalized_sample_landmarks[batch_n][:, 0] = non_normalized_sample_landmarks[batch_n][:, 0] / meta['scale'][0]
            non_normalized_sample_landmarks[batch_n][:, 1] = non_normalized_sample_landmarks[batch_n][:, 1] / meta['scale'][1]


        print('debug 2 ', non_normalized_output, '    ', non_normalized_sample_landmarks)'''
        normalized_output = copy.copy(non_normalized_output)
        normalized_sample_landmarks = copy.copy(non_normalized_sample_landmarks)
        print('debug landmarks ', normalized_output[0], '   ', normalized_sample_landmarks[0])
        print('debug landmarks ', normalized_output[1], '   ', normalized_sample_landmarks[1])

        '''for normalize_idx in range(batch_size):
            h = args.metas[normalize_idx]['width_height'][1]
            w = args.metas[normalize_idx]['width_height'][0]
            a = [float(w), float(h)]
            a = np.expand_dims(a, axis = 1)
            b = np.concatenate([a.T for _ in range(8)], axis = 0)
            normalized_output[normalize_idx] = normalized_output[normalize_idx] / b
            normalized_sample_landmarks[normalize_idx] = normalized_sample_landmarks[normalize_idx] / b'''

        h, w, _ = args.images[0].shape
        print('to compare ', args.metas[0]['width_height'], '  ', h)

        a = [float(w), float(h)]
        a = np.expand_dims(a, axis = 1)
        b = np.concatenate([a.T for i in range(8)], axis = 0)
        c = np.stack([b for _ in range(batch_size)], axis = 0)

        normalized_output = normalized_output / c
        normalized_sample_landmarks = normalized_sample_landmarks / c

        for normalize_idx in range(batch_size):
            h_proportion = args.metas[normalize_idx]['height_proportion']
            w_proportion = args.metas[normalize_idx]['width_proportion']
            a = [float(w_proportion), float(h_proportion)]
            a = np.expand_dims(a, axis = 1)
            b = np.concatenate([a.T for _ in range(8)], axis = 0)
            normalized_output[normalize_idx] = normalized_output[normalize_idx] * b
            normalized_sample_landmarks[normalize_idx] = normalized_sample_landmarks[normalize_idx] * b
           


        # MAKING EACH VISIBILITY TWO COLUMNS
        dist_batch_lm_vis_count_all = torch.from_numpy(batch_lm_vis_count_all).float()
        dist_batch_lm_vis_count_all = torch.cat([dist_batch_lm_vis_count_all, dist_batch_lm_vis_count_all], dim=2).cpu().detach().numpy()

        dist_batch_lm_pic_count_all = torch.from_numpy(batch_lm_pic_count_all).float()
        dist_batch_lm_pic_count_all = torch.cat([dist_batch_lm_pic_count_all, dist_batch_lm_pic_count_all], dim=2).cpu().detach().numpy()

        dist_batch_lm_vis_count_intersection = torch.from_numpy(batch_lm_vis_count_intersection).float()
        dist_batch_lm_vis_count_intersection = torch.cat([dist_batch_lm_vis_count_intersection, dist_batch_lm_vis_count_intersection], dim=2).cpu().detach().numpy()

        dist_batch_lm_pic_count_intersection = torch.from_numpy(batch_lm_pic_count_intersection).float()
        dist_batch_lm_pic_count_intersection = torch.cat([dist_batch_lm_pic_count_intersection, dist_batch_lm_pic_count_intersection], dim=2).cpu().detach().numpy()

    

        # COPUTING THE DISTANCES WITH THE GIVEN 
        batch_lm_vis_dist = np.sum(np.sqrt(np.sum(np.square(dist_batch_lm_vis_count_all * normalized_output - dist_batch_lm_vis_count_all * normalized_sample_landmarks,), axis=2)), axis=0).reshape((8, 1))
        batch_lm_pic_dist = np.sum(np.sqrt(np.sum(np.square(dist_batch_lm_pic_count_all * normalized_output - dist_batch_lm_pic_count_all * normalized_sample_landmarks,), axis=2)), axis=0).reshape((8, 1))
        batch_lm_vis_dist_intersection = np.sum(np.sqrt(np.sum(np.square(dist_batch_lm_vis_count_intersection * normalized_output - dist_batch_lm_vis_count_intersection * normalized_sample_landmarks,), axis=2)), axis=0).reshape((8, 1)) 
        batch_lm_pic_dist_intersection = np.sum(np.sqrt(np.sum(np.square(dist_batch_lm_pic_count_intersection * normalized_output - dist_batch_lm_pic_count_intersection * normalized_sample_landmarks,), axis=2)), axis=0).reshape((8, 1)) 
        
        '''for batch_n in range(batch_size):
            batch_dist_vis = batch_separated_vis_distances[batch_n]
            batch_dist_pic = batch_separated_pic_distances[batch_n]

            temp_dict = {}
            temp_dict['gt_landmarks'] = np.array(non_normalized_sample_landmarks[batch_n])
            temp_dict['gt_vis'] = np.array(batch_lm_vis_count_all[batch_n])
            temp_dict['pred_landmarks'] = np.array(non_normalized_output[batch_n])
            for landmark_n in range(8):
                temp_dict['vis'+str(landmark_n)] = batch_dist_vis[landmark_n]
            temp_dict['vis_mean'] = batch_dist_vis.mean()
            for landmark_n in range(8):
                temp_dict['pic'+str(landmark_n)] = batch_dist_pic[landmark_n]
            temp_dict['pic_mean'] = batch_dist_pic.mean()

            # join this latter dictionary with the meta of the image

            current_meta_dict = args.metas[batch_n]
            total_batch_dict = {**current_meta_dict, **temp_dict}

            csv_columns = total_batch_dict.keys()

            try:
                with open(args.tensorboard_dir+'.csv', 'a') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                    if args.first_round:
                        writer.writeheader()
                        args.first_round = False
                    writer.writerow(total_batch_dict)
            except IOError:
                print("I/O error")
            
            print('just before the error ', type(temp_dict))
            with open("images_info.json", "w") as outfile:
                json.dump(total_batch_dict, outfile)'''



        self.lm_vis_dist += batch_lm_vis_dist
        print('error in this batch ', batch_lm_vis_dist/np.sum(batch_lm_vis_count_all, axis=0))
        self.lm_pic_dist += batch_lm_pic_dist
        self.lm_vis_dist_intersection += batch_lm_vis_dist_intersection
        self.lm_pic_dist_intersection += batch_lm_pic_dist_intersection
       

    def add(self, output, visib_preds, sample):
        self.landmark_count(output, visib_preds, sample)

    def evaluate(self):


        return {
            'individual_lm_vis_dist' : (self.lm_vis_dist/self.lm_vis_count_all),
            'lm_vis_dist' : (self.lm_vis_dist/self.lm_vis_count_all).mean(),
            'individual_lm_pic_dist' : (self.lm_pic_dist/self.lm_pic_count_all),
            'lm_pic_dist' : (self.lm_pic_dist/self.lm_pic_count_all).mean(),
            'individual_lm_vis_dist_intersection' : (self.lm_vis_dist_intersection/self.lm_vis_count_intersection),
            'lm_vis_dist_intersection' : (self.lm_vis_dist_intersection/self.lm_vis_count_intersection).mean(),
            'individual_lm_pic_dist_intersection' : (self.lm_pic_dist_intersection/self.lm_pic_count_intersection),
            'lm_pic_dist_intersection' : (self.lm_pic_dist_intersection/self.lm_pic_count_intersection).mean(),
            'in_pred_not_gt' : (self.in_pred_not_gt/self.n_instances),
            'in_gt_not_pred' : (self.in_gt_not_pred/self.n_instances),
            'in_pic_not_pred' : (self.in_pic_not_pred/self.n_instances),
        }


# so we are just considering the visibility of the gt, but sometimes some points are indicated visible in gt
# and are not visible in the prediction. so maybe pass two visibilities.
def kp_visualizer(temp_image_name, np_image, vis_pred, vis_gt, pred_lms, gt_lms):


    with image_canvas(np_image, show=False, fig_file = temp_image_name + '_pred.png', fig_width=10, dpi_factor=1.0) as ax:
                
        keypoints(ax, vis_pred, pred_lms, 'go', size=np_image.size)

    with image_canvas(np_image, show=False, fig_file= temp_image_name + '_gt.png', fig_width=10, dpi_factor=1.0) as ax:

        keypoints(ax, vis_gt, gt_lms, 'bo', size=np_image.size)
    
    image_output1 = Image.open(temp_image_name+ '_pred.png').convert('RGB')
    output1 = np.asarray(image_output1)


    image_output2 = Image.open(temp_image_name+ '_gt.png').convert('RGB')
    output2 = np.asarray(image_output2)




    return output1, output2
    
    

           
def keypoints(ax, vis, lms, differentizer_color,  *,
              size=None, scores=None, color=None,
              colors=None, texts=None, activities=None, dic_out=None):
    
    
    if lms is None:
        return

    if color is None and colors is None:
        colors = range(len(lms))

    # we only have one set of keypoints here
    for i, (vis, lm) in enumerate(zip(np.asarray(vis), np.asarray(lms))):
        assert lm.shape[1] == 2
        # assert len(vis) == 1
        xy_scale = 1
        y_scale = 1
        x = lm[:, 0] * xy_scale
        y = lm[:, 1] * xy_scale * y_scale
        v = vis[:, 0] # visibility

       
        if isinstance(color, (int, np.integer)):
            color = matplotlib.cm.get_cmap('tab20')((color % 20 + 0.05) / 20)

        _draw_skeleton(ax, x, y, v, differentizer_color,  i=i, size=size, color=color, activities=activities, dic_out=dic_out)

        if texts is not None:
            draw_text(ax, x, y, v, texts[i], color)

       
def _draw_skeleton(ax, x, y, v, differentizer_color,  *, i=0, size=None, color=None, activities=None, dic_out=None):
        
    if not np.any(v > 0):
        return
    for n_joint, (joint_x, joint_y) in enumerate(zip(x, y)):
        if v[n_joint] > 0:
            c = color
            ax.plot(joint_x, joint_y, differentizer_color)

    if DEEPFASHION_SKELETON is not None:
        for ci, connection in enumerate(np.array(DEEPFASHION_SKELETON) - 1):
            # cnnection is of form [a, b]
            c = color
            linewidth = 2

            color_connections = False
            dashed_threshold = 0.005
            solid_threshold = 0.005
            if color_connections:
                c = matplotlib.cm.get_cmap('tab20')(ci / len(DEEPFASHION_SKELETON))
            if np.all(v[connection] > dashed_threshold):
                ax.plot(x[connection], y[connection],
                        linewidth=linewidth, color=c,
                        linestyle='dashed', dash_capstyle='round')
            if np.all(v[connection] > solid_threshold):
                ax.plot(x[connection], y[connection],
                        linewidth=linewidth, color=c, solid_capstyle='round')
        


  
@contextmanager
def image_canvas(image, fig_file=None, show=True, dpi_factor=1.0, fig_width=10.0, **kwargs):
    if 'figsize' not in kwargs:
        kwargs['figsize'] = (fig_width, fig_width * image.shape[1] / image.shape[0])

    if ndimage is None:
        raise Exception('please install scipy')
    fig = plt.figure(**kwargs)
    canvas = FigureCanvas(fig)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    ax.set_xlim(0, image.shape[0])
    ax.set_ylim(image.shape[1], 0)
    fig.add_axes(ax)
    image_2 = ndimage.gaussian_filter(image, sigma=2.5)
    ax.imshow(image_2, alpha=0.4)
    yield ax

    canvas.draw()       # draw the canvas, cache the renderer
    imaged_plot = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    # this image should be sent to running code part to be added to tensorboard
    if fig_file:
        fig.savefig(fig_file, dpi=image.shape[0] / kwargs['figsize'][0] * dpi_factor)
        print('keypoints image saved')

    
    plt.close(fig)


def draw_text(ax, x, y, v, text, color, fontsize=8):
    if not np.any(v > 0):
        return

    # keypoint bounding box
    x1, x2 = np.min(x[v > 0]), np.max(x[v > 0])
    y1, y2 = np.min(y[v > 0]), np.max(y[v > 0])
    if x2 - x1 < 5.0:
        x1 -= 2.0
        x2 += 2.0
    if y2 - y1 < 5.0:
        y1 -= 2.0
        y2 += 2.0

    ax.text(x1 + 2, y1 - 2, text, fontsize=fontsize,
            color='white', bbox={'facecolor': color, 'alpha': 0.5, 'linewidth': 0})

