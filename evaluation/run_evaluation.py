# the goal is to 1. have a json of the gt and pred 2. obtain some metrics on the predictions
from calendar import EPOCH
from datamodule_deepfashion import eval_loader
import pandas as pd
import torch
import torch.utils.data
# from tensorboardX import SummaryWriter
from predict_evaluation import pifpaf_output, LandmarkEvaluator
import argparse
from openpifpaf import decoder, network, visualizer, show, logger
import os
import random
import numpy as np
import csv
import json
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision


def name_constructor(n_epoch):

    if n_epoch >= 100:
        return str(n_epoch)
    elif n_epoch < 10:
        return '00'+str(n_epoch)
    else:
        return '0'+str(n_epoch)
        

def cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = parser.add_subparsers(help='Different parsers for main actions', dest='command')
    predict_parser = subparsers.add_parser("predict")
    # Predict (2D pose and/or 3D location from images)
    predict_parser.add_argument('--eval-image-dir', help='where the eval images are stored')
    predict_parser.add_argument('--eval-annotations', help='where the eval annotations are stored')
    predict_parser.add_argument('--batch_size', default=20, type=int, help='batch size for evaluation, max recommended is 40')
    predict_parser.add_argument('--single-epoch', default=False, type=bool, help='only one epoch to evaluatie and print the results')
    predict_parser.add_argument('--confidence_thresh', default=0.1, type=float, help='the threshhold for the confidence to be considered as visible in the prediction')
    predict_parser.add_argument('--force_complete_pose', default=True, help='whether all the landmark should be forced to be detected', type=bool)
    predict_parser.add_argument('--tensorboard_dir', help='where to write the logs to visualize in tensorboard')
    predict_parser.add_argument('--checkpoint_name_1', help='pifpaf model')
    predict_parser.add_argument('--checkpoint_name_2', help='pifpaf model')
    predict_parser.add_argument('--temp_image_name', help='image name for keeping the annotated image temporarily', default='sibzamini.png', type=str)
    predict_parser.add_argument('--ckpt_directory', default='/Users/pegahkhayatan/Desktop/deepfashion_project/models/' ,help='directory where to find all the trained models')
    predict_parser.add_argument('--jump', type = int, help='skip how many epochs before evaluating again')
    predict_parser.add_argument('--start_epoch_1', type = int, help='where to start the evaluation')
    predict_parser.add_argument('--start_epoch_2', type = int, help='where to start the evaluation')
    predict_parser.add_argument('--n_epochs_1', type = int, help='total number of trained epochs')
    predict_parser.add_argument('--n_epochs_2', type = int, help='total number of trained epochs')
    predict_parser.add_argument('--n_sample', type = int, help='number of samples for evaluation in every checkpoint')
   
    predict_parser.add_argument('--dpi', help='image resolution', type=int, default=100)
    predict_parser.add_argument('--long-edge', default=None, type=int,
                                help='rescale the long side of the image (aspect ratio maintained)')
    
    
    predict_parser.add_argument('--disable-cuda', action='store_true', help='disable CUDA')
    
    predict_parser.add_argument('--decoder-workers', default=None, type=int,
                                help='number of workers for pose decoding, 0 for windows')

    predict_parser.add_argument('--seed-threshold', type=float, default=0.5, help='threshold for single seed')
    predict_parser.add_argument('--precise-rescaling', dest='fast_rescaling', default=True, action='store_false',
                                help='use more exact image rescaling (requires scipy)')
   
    decoder.cli(parser)
    logger.cli(parser)
    network.Factory.cli(parser)
    show.cli(parser)
    visualizer.cli(parser)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = cli()
    test_dataloader = eval_loader(args)
    
    checkpoint_name = args.checkpoint_name_1
    n_epochs = args.n_epochs_1
    jump = args.jump
    start_epoch = args.start_epoch_1
   
    if not args.single_epoch:
        assert args.tensorboard_dir is not None, 'no directory is given to write tensorboard logs'
        writer = SummaryWriter(str(args.tensorboard_dir))

    evaluator = LandmarkEvaluator()
    
    for current_epoch in np.arange(start_epoch, n_epochs, jump):

        print('pre-checking epochs ', np.arange(start_epoch, n_epochs, jump))
        annotated_imgs = []

        
        evaluator.reset()
        if not args.single_epoch:
            args.checkpoint = args.ckpt_directory + checkpoint_name + name_constructor(current_epoch)
        else:
            args.checkpoint = args.ckpt_directory + checkpoint_name


        args.checkpoint_name = checkpoint_name
        print('epoch {} checkpoint is {}'.format(current_epoch, args.checkpoint))
        counter_visualization = 0
        args.first_round = True

        for _, (images, anns, metas) in enumerate(test_dataloader):
            batch_size = images.shape[0]
            counter_visualization += 1
            print('the current ckpt ', args.checkpoint, '    ', images.shape)
            mean_data = torch.mean(images)
            std_data = torch.std(images)
            images = (images - mean_data)/std_data
            images = ((images-torch.min(images))*255.0)/(torch.max(images)-torch.min(images))
            images = images.int()
            
            
            args.images = np.transpose(images.cpu().detach().numpy().astype(np.uint8), (0, 2, 3, 1))
            args.metas = metas

            #gt_anns = [ann[0].inverse_transform(metas[idx_anns]) for idx_anns, ann in enumerate(anns)]

            keypoints = np.array([anns[i][0]['keypoints'] for i in range(batch_size)]).reshape((batch_size, -1))

            landmarks = np.array([keypoints[k][3*i+j] for k in range(batch_size) for i in range(8) for j in range(2)]).reshape((batch_size, 8, 2))
            visibilities = np.array([keypoints[k][3*i+2] for k in range(batch_size) for i in range(8)]).reshape((batch_size, 8, 1))
            args.landmarks = landmarks
            args.vis =  visibilities

            '''if (counter_visualization % 40 == 0 and not args.single_epoch):
                lm_output, visib_preds, annotated_img_pred, annotated_img_gt = pifpaf_output(args, visualize = True)
                annotated_imgs.append(annotated_img_pred)
                annotated_imgs.append(annotated_img_gt)
            else:'''
            lm_output, visib_preds = pifpaf_output(args, visualize = False)


                       
            evaluator.add(lm_output, visib_preds, args)
            
        dict_eval = evaluator.evaluate()


        '''
            'individual_lm_vis_dist' : (self.lm_vis_dist/self.lm_vis_count_all),
            'lm_vis_dist' : (self.lm_vis_dist/self.lm_vis_count_all).mean(),
            'individual_lm_pic_dist' : (self.lm_pic_dist/self.lm_pic_dist),
            'lm_pic_dist' : (self.lm_pic_dist/self.lm_pic_dist).mean(),
            'individual_lm_vis_dist_intersection' : (self.lm_vis_dist_intersection/self.lm_vis_count_intersection),
            'lm_vis_dist_intersection' : (self.lm_vis_dist_intersection/self.lm_vis_count_intersection).mean(),
            'individual_lm_pic_dist_intersection' : (self.lm_pic_dist_intersection/self.lm_pic_count_intersection),
            'lm_pic_dist_intersection' : (self.lm_pic_dist_intersection/self.lm_pic_count_intersection).mean(),
            'individual_lm_vis_dist_intersection_lowconf' : (self.lm_vis_dist_intersection_lowconf/self.lm_vis_count_intersection_lowconf),
            'lm_vis_dist_intersection_lowconf' : (self.lm_vis_dist_intersection_lowconf/self.lm_vis_count_intersection_lowconf).mean(),
            'individual_lm_pic_dist_intersection_lowconf' : (self.lm_pic_dist_intersection_lowconf/self.lm_pic_count_intersection_lowconf),
            'lm_pic_dist_intersection_lowconf' : (self.lm_pic_dist_intersection_lowconf/self.lm_pic_count_intersection_lowconf).mean(),
            'in_pred_not_gt' : (self.in_pred_not_gt/self.n_instances),
            'in_gt_not_pred' : (self.in_gt_not_pred/self.n_instances),
            'in_pic_not_pred' : (self.in_pic_not_pred/self.n_instances),
            'in_pic_not_gt' : (self.in_pic_not_gt/self.n_instances)
        '''

        if args.single_epoch:
            print('the average NE over vis in gt is {}'.format( dict_eval['lm_vis_dist']))
            print('the individual NE over vis in gt is {}'.format( dict_eval['individual_lm_vis_dist']))


            print('the average NE over pic in gt is {}'.format( dict_eval['lm_pic_dist']))
            print('the individual NE over pic in gt is {}'.format( dict_eval['individual_lm_pic_dist']))


            print('the average NE over vis intersection in gt is {}'.format( dict_eval['lm_vis_dist_intersection']))
            print('the individual NE over vis intersection in gt is {}'.format( dict_eval['individual_lm_vis_dist_intersection']))


            print('the average NE over pic intersection in gt is {}'.format( dict_eval['lm_pic_dist_intersection']))
            print('the individual NE over pic intersection in gt is {}'.format( dict_eval['individual_lm_pic_dist_intersection']))

            print('percentage of points in pred not gt {}'.format(dict_eval['in_pred_not_gt']))
            print('percentage of points in gt not pred {}'.format(dict_eval['in_gt_not_pred']))
            print('percentage of points in pic not pred {}'.format(dict_eval['in_pic_not_pred']))

            quit()




        writer.add_scalar('VIS_NE/AVG', dict_eval['lm_vis_dist'], current_epoch)
        print('the average NE over vis in gt is {}'.format( dict_eval['lm_vis_dist']))
        print('the individual NE over vis in gt is {}'.format( dict_eval['individual_lm_vis_dist']))
        writer.add_scalar('VIS_NE/L_col', dict_eval['individual_lm_vis_dist'][0], current_epoch)
        writer.add_scalar('VIS_NE/R_col', dict_eval['individual_lm_vis_dist'][1], current_epoch)
        writer.add_scalar('VIS_NE/L_Sle', dict_eval['individual_lm_vis_dist'][2], current_epoch)
        writer.add_scalar('VIS_NE/R_Sle', dict_eval['individual_lm_vis_dist'][3], current_epoch)
        writer.add_scalar('VIS_NE/L_Wai', dict_eval['individual_lm_vis_dist'][4], current_epoch)
        writer.add_scalar('VIS_NE/R_Wai', dict_eval['individual_lm_vis_dist'][5], current_epoch)
        writer.add_scalar('VIS_NE/L_Hem', dict_eval['individual_lm_vis_dist'][6], current_epoch)
        writer.add_scalar('VIS_NE/R_Hem', dict_eval['individual_lm_vis_dist'][7], current_epoch)



        writer.add_scalar('PIC_NE/AVG', dict_eval['lm_pic_dist'], current_epoch)
        writer.add_scalar('PIC_NE/L_col', dict_eval['individual_lm_pic_dist'][0], current_epoch)
        writer.add_scalar('PIC_NE/R_col', dict_eval['individual_lm_pic_dist'][1], current_epoch)
        writer.add_scalar('PIC_NE/L_Sle', dict_eval['individual_lm_pic_dist'][2], current_epoch)
        writer.add_scalar('PIC_NE/R_Sle', dict_eval['individual_lm_pic_dist'][3], current_epoch)
        writer.add_scalar('PIC_NE/L_Wai', dict_eval['individual_lm_pic_dist'][4], current_epoch)
        writer.add_scalar('PIC_NE/R_Wai', dict_eval['individual_lm_pic_dist'][5], current_epoch)
        writer.add_scalar('PIC_NE/L_Hem', dict_eval['individual_lm_pic_dist'][6], current_epoch)
        writer.add_scalar('PIC_NE/R_Hem', dict_eval['individual_lm_pic_dist'][7], current_epoch)



        writer.add_scalar('VIS_INTER/AVG', dict_eval['lm_vis_dist_intersection'], current_epoch)
        writer.add_scalar('VIS_INTER/L_col', dict_eval['individual_lm_vis_dist_intersection'][0], current_epoch)
        writer.add_scalar('VIS_INTER/R_col', dict_eval['individual_lm_vis_dist_intersection'][1], current_epoch)
        writer.add_scalar('VIS_INTER/L_Sle', dict_eval['individual_lm_vis_dist_intersection'][2], current_epoch)
        writer.add_scalar('VIS_INTER/R_Sle', dict_eval['individual_lm_vis_dist_intersection'][3], current_epoch)
        writer.add_scalar('VIS_INTER/L_Wai', dict_eval['individual_lm_vis_dist_intersection'][4], current_epoch)
        writer.add_scalar('VIS_INTER/R_Wai', dict_eval['individual_lm_vis_dist_intersection'][5], current_epoch)
        writer.add_scalar('VIS_INTER/L_Hem', dict_eval['individual_lm_vis_dist_intersection'][6], current_epoch)
        writer.add_scalar('VIS_INTER/R_Hem', dict_eval['individual_lm_vis_dist_intersection'][7], current_epoch)


        writer.add_scalar('PIC_INTER/AVG', dict_eval['lm_pic_dist_intersection'], current_epoch)
        writer.add_scalar('PIC_INTER/L_col', dict_eval['individual_lm_pic_dist_intersection'][0], current_epoch)
        writer.add_scalar('PIC_INTER/R_col', dict_eval['individual_lm_pic_dist_intersection'][1], current_epoch)
        writer.add_scalar('PIC_INTER/L_Sle', dict_eval['individual_lm_pic_dist_intersection'][2], current_epoch)
        writer.add_scalar('PIC_INTER/R_Sle', dict_eval['individual_lm_pic_dist_intersection'][3], current_epoch)
        writer.add_scalar('PIC_INTER/L_Wai', dict_eval['individual_lm_pic_dist_intersection'][4], current_epoch)
        writer.add_scalar('PIC_INTER/R_Wai', dict_eval['individual_lm_pic_dist_intersection'][5], current_epoch)
        writer.add_scalar('PIC_INTER/L_Hem', dict_eval['individual_lm_pic_dist_intersection'][6], current_epoch)
        writer.add_scalar('PIC_INTER/R_Hem', dict_eval['individual_lm_pic_dist_intersection'][7], current_epoch)

        
        # DETECTED BUT NOT IN GT FOR EACH TYPE OF LANDMARK
        writer.add_scalar('PRED_N_GT/L_col', dict_eval['in_pred_not_gt'][0], current_epoch)
        writer.add_scalar('PRED_N_GT/R_col', dict_eval['in_pred_not_gt'][1], current_epoch)
        writer.add_scalar('PRED_N_GT/L_Sle', dict_eval['in_pred_not_gt'][2], current_epoch)
        writer.add_scalar('PRED_N_GT/R_Sle', dict_eval['in_pred_not_gt'][3], current_epoch)
        writer.add_scalar('PRED_N_GT/L_Wai', dict_eval['in_pred_not_gt'][4], current_epoch)
        writer.add_scalar('PRED_N_GT/R_Wai', dict_eval['in_pred_not_gt'][5], current_epoch)
        writer.add_scalar('PRED_N_GT/L_Hem', dict_eval['in_pred_not_gt'][6], current_epoch)
        writer.add_scalar('PRED_N_GT/R_Hem', dict_eval['in_pred_not_gt'][7], current_epoch)



        writer.add_scalar('GT_N_PRED/L_col', dict_eval['in_gt_not_pred'][0], current_epoch)
        writer.add_scalar('GT_N_PRED/R_col', dict_eval['in_gt_not_pred'][1], current_epoch)
        writer.add_scalar('GT_N_PRED/L_Sle', dict_eval['in_gt_not_pred'][2], current_epoch)
        writer.add_scalar('GT_N_PRED/R_Sle', dict_eval['in_gt_not_pred'][3], current_epoch)
        writer.add_scalar('GT_N_PRED/L_Wai', dict_eval['in_gt_not_pred'][4], current_epoch)
        writer.add_scalar('GT_N_PRED/R_Wai', dict_eval['in_gt_not_pred'][5], current_epoch)
        writer.add_scalar('GT_N_PRED/L_Hem', dict_eval['in_gt_not_pred'][6], current_epoch)
        writer.add_scalar('GT_N_PRED/R_Hem', dict_eval['in_gt_not_pred'][7], current_epoch)

        writer.add_scalar('PIC_N_PRED/L_col', dict_eval['in_pic_not_pred'][0], current_epoch)
        writer.add_scalar('PIC_N_PRED/R_col', dict_eval['in_pic_not_pred'][1], current_epoch)
        writer.add_scalar('PIC_N_PRED/L_Sle', dict_eval['in_pic_not_pred'][2], current_epoch)
        writer.add_scalar('PIC_N_PRED/R_Sle', dict_eval['in_pic_not_pred'][3], current_epoch)
        writer.add_scalar('PIC_N_PRED/L_Wai', dict_eval['in_pic_not_pred'][4], current_epoch)
        writer.add_scalar('PIC_N_PRED/R_Wai', dict_eval['in_pic_not_pred'][5], current_epoch)
        writer.add_scalar('PIC_N_PRED/L_Hem', dict_eval['in_pic_not_pred'][6], current_epoch)
        writer.add_scalar('PIC_N_PRED/R_Hem', dict_eval['in_pic_not_pred'][7], current_epoch)


        tensored_annotated_imgs = torch.tensor(np.array(annotated_imgs))
        print('all the information ', len(tensored_annotated_imgs.shape), '   ', tensored_annotated_imgs.shape)
        #writer.add_images('images_'+str(current_epoch), tensored_annotated_imgs,current_epoch, dataformats = 'NHWC')




    if args.checkpoint_name_2 is not None:

        checkpoint_name = args.checkpoint_name_2 # resnet18-220422-164833-deepfashion-slurm940492.pkl.epoch ????

        
        n_epochs = int(args.n_epochs_2)
        start_epoch = int(args.start_epoch_2)


        for i in np.arange(start_epoch, n_epochs, jump):

            print('the epoch ', i)
            annotated_imgs = []

            evaluator.reset()
            args.checkpoint = args.ckpt_directory + checkpoint_name + name_constructor(i)
            args.checkpoint_name = checkpoint_name+str(i)
            print('epoch {} checkpoint is {}'.format(i, args.checkpoint))
            counter_visualization = 0
            for j, sample in enumerate(test_dataloader): 
                counter_visualization += 1
                print('the current ckpt ', args.checkpoint)

                args.images = images.cpu().detach().numpy()
                keypoints = anns['keypoints']
                landmarks = np.array([keypoints[3*i+j] for i in range(8) for j in range(2)])
                visibilities = np.array([keypoints[3*i+2] for i in range(8)])
                args.landmarks = landmarks
                args.vis =  visibilities

                if counter_visualization % 40 == 0:
                    lm_output, visib_preds, annotated_img_pred, annotated_img_gt = pifpaf_output(args, visualize = True)
                    annotated_imgs.append(annotated_img_pred)
                    annotated_imgs.append(annotated_img_gt)
                else:
                    lm_output, visib_preds = pifpaf_output(args, visualize = False)

                evaluator.add(lm_output, visib_preds, args)

            dict_eval = evaluator.evaluate()
            
            '''
                'individual_lm_vis_dist' : (self.lm_vis_dist/self.lm_vis_count_all),
                'lm_vis_dist' : (self.lm_vis_dist/self.lm_vis_count_all).mean(),
                'individual_lm_pic_dist' : (self.lm_pic_dist/self.lm_pic_dist),
                'lm_pic_dist' : (self.lm_pic_dist/self.lm_pic_dist).mean(),
                'individual_lm_vis_dist_intersection' : (self.lm_vis_dist_intersection/self.lm_vis_count_intersection),
                'lm_vis_dist_intersection' : (self.lm_vis_dist_intersection/self.lm_vis_count_intersection).mean(),
                'individual_lm_pic_dist_intersection' : (self.lm_pic_dist_intersection/self.lm_pic_count_intersection),
                'lm_pic_dist_intersection' : (self.lm_pic_dist_intersection/self.lm_pic_count_intersection).mean(),
                'individual_lm_vis_dist_intersection_lowconf' : (self.lm_vis_dist_intersection_lowconf/self.lm_vis_count_intersection_lowconf),
                'lm_vis_dist_intersection_lowconf' : (self.lm_vis_dist_intersection_lowconf/self.lm_vis_count_intersection_lowconf).mean(),
                'individual_lm_pic_dist_intersection_lowconf' : (self.lm_pic_dist_intersection_lowconf/self.lm_pic_count_intersection_lowconf),
                'lm_pic_dist_intersection_lowconf' : (self.lm_pic_dist_intersection_lowconf/self.lm_pic_count_intersection_lowconf).mean(),
                'in_pred_not_gt' : (self.in_pred_not_gt/self.n_instances),
                'in_gt_not_pred' : (self.in_gt_not_pred/self.n_instances),
                'in_pic_not_pred' : (self.in_pic_not_pred/self.n_instances),
                'in_pic_not_gt' : (self.in_pic_not_gt/self.n_instances)
            '''
            print('the average NE over vis in gt is {}'.format( dict_eval['lm_dist']))
            print('the individual NE over vis in gt is {}'.format( dict_eval['lm_individual_dist']))
            writer.add_scalar('VIS_NE/AVG', dict_eval['lm_vis_dist'], i)
            writer.add_scalar('VIS_NE/L_col', dict_eval['individual_lm_vis_dist'][0], i)
            writer.add_scalar('VIS_NE/R_col', dict_eval['individual_lm_vis_dist'][1], i)
            writer.add_scalar('VIS_NE/L_Sle', dict_eval['individual_lm_vis_dist'][2], i)
            writer.add_scalar('VIS_NE/R_Sle', dict_eval['individual_lm_vis_dist'][3], i)
            writer.add_scalar('VIS_NE/L_Wai', dict_eval['individual_lm_vis_dist'][4], i)
            writer.add_scalar('VIS_NE/R_Wai', dict_eval['individual_lm_vis_dist'][5], i)
            writer.add_scalar('VIS_NE/L_Hem', dict_eval['individual_lm_vis_dist'][6], i)
            writer.add_scalar('VIS_NE/R_Hem', dict_eval['individual_lm_vis_dist'][7], i)



            writer.add_scalar('PIC_NE/AVG', dict_eval['lm_pic_dist'], i)
            writer.add_scalar('PIC_NE/L_col', dict_eval['individual_lm_pic_dist'][0], i)
            writer.add_scalar('PIC_NE/R_col', dict_eval['individual_lm_pic_dist'][1], i)
            writer.add_scalar('PIC_NE/L_Sle', dict_eval['individual_lm_pic_dist'][2], i)
            writer.add_scalar('PIC_NE/R_Sle', dict_eval['individual_lm_pic_dist'][3], i)
            writer.add_scalar('PIC_NE/L_Wai', dict_eval['individual_lm_pic_dist'][4], i)
            writer.add_scalar('PIC_NE/R_Wai', dict_eval['individual_lm_pic_dist'][5], i)
            writer.add_scalar('PIC_NE/L_Hem', dict_eval['individual_lm_pic_dist'][6], i)
            writer.add_scalar('PIC_NE/R_Hem', dict_eval['individual_lm_pic_dist'][7], i)



            writer.add_scalar('VIS_INTER/AVG', dict_eval['lm_vis_dist_intersection'], i)
            writer.add_scalar('VIS_INTER/L_col', dict_eval['individual_lm_vis_dist_intersection'][0], i)
            writer.add_scalar('VIS_INTER/R_col', dict_eval['individual_lm_vis_dist_intersection'][1], i)
            writer.add_scalar('VIS_INTER/L_Sle', dict_eval['individual_lm_vis_dist_intersection'][2], i)
            writer.add_scalar('VIS_INTER/R_Sle', dict_eval['individual_lm_vis_dist_intersection'][3], i)
            writer.add_scalar('VIS_INTER/L_Wai', dict_eval['individual_lm_vis_dist_intersection'][4], i)
            writer.add_scalar('VIS_INTER/R_Wai', dict_eval['individual_lm_vis_dist_intersection'][5], i)
            writer.add_scalar('VIS_INTER/L_Hem', dict_eval['individual_lm_vis_dist_intersection'][6], i)
            writer.add_scalar('VIS_INTER/R_Hem', dict_eval['individual_lm_vis_dist_intersection'][7], i)


            writer.add_scalar('PIC_INTER/AVG', dict_eval['lm_pic_dist_intersection'], i)
            writer.add_scalar('PIC_INTER/L_col', dict_eval['individual_lm_pic_dist_intersection'][0], i)
            writer.add_scalar('PIC_INTER/R_col', dict_eval['individual_lm_pic_dist_intersection'][1], i)
            writer.add_scalar('PIC_INTER/L_Sle', dict_eval['individual_lm_pic_dist_intersection'][2], i)
            writer.add_scalar('PIC_INTER/R_Sle', dict_eval['individual_lm_pic_dist_intersection'][3], i)
            writer.add_scalar('PIC_INTER/L_Wai', dict_eval['individual_lm_pic_dist_intersection'][4], i)
            writer.add_scalar('PIC_INTER/R_Wai', dict_eval['individual_lm_pic_dist_intersection'][5], i)
            writer.add_scalar('PIC_INTER/L_Hem', dict_eval['individual_lm_pic_dist_intersection'][6], i)
            writer.add_scalar('PIC_INTER/R_Hem', dict_eval['individual_lm_pic_dist_intersection'][7], i)


            writer.add_scalar('LOWCONF_VIS/AVG', dict_eval['lm_vis_dist_intersection_lowconf'], i)
            writer.add_scalar('LOWCONF_VIS/L_col', dict_eval['individual_lm_vis_dist_intersection_lowconf'][0], i)
            writer.add_scalar('LOWCONF_VIS/R_col', dict_eval['individual_lm_vis_dist_intersection_lowconf'][1], i)
            writer.add_scalar('LOWCONF_VIS/L_Sle', dict_eval['individual_lm_vis_dist_intersection_lowconf'][2], i)
            writer.add_scalar('LOWCONF_VIS/R_Sle', dict_eval['individual_lm_vis_dist_intersection_lowconf'][3], i)
            writer.add_scalar('LOWCONF_VIS/L_Wai', dict_eval['individual_lm_vis_dist_intersection_lowconf'][4], i)
            writer.add_scalar('LOWCONF_VIS/R_Wai', dict_eval['individual_lm_vis_dist_intersection_lowconf'][5], i)
            writer.add_scalar('LOWCONF_VIS/L_Hem', dict_eval['individual_lm_vis_dist_intersection_lowconf'][6], i)
            writer.add_scalar('LOWCONF_VIS/R_Hem', dict_eval['individual_lm_vis_dist_intersection_lowconf'][7], i)


            writer.add_scalar('LOWCONF_PIC/AVG', dict_eval['lm_pic_dist_intersection_lowconf'], i)
            writer.add_scalar('LOWCONF_PIC/L_col', dict_eval['individual_lm_pic_dist_intersection_lowconf'][0], i)
            writer.add_scalar('LOWCONF_PIC/R_col', dict_eval['individual_lm_pic_dist_intersection_lowconf'][1], i)
            writer.add_scalar('LOWCONF_PIC/L_Sle', dict_eval['individual_lm_pic_dist_intersection_lowconf'][2], i)
            writer.add_scalar('LOWCONF_PIC/R_Sle', dict_eval['individual_lm_pic_dist_intersection_lowconf'][3], i)
            writer.add_scalar('LOWCONF_PIC/L_Wai', dict_eval['individual_lm_pic_dist_intersection_lowconf'][4], i)
            writer.add_scalar('LOWCONF_PIC/R_Wai', dict_eval['individual_lm_pic_dist_intersection_lowconf'][5], i)
            writer.add_scalar('LOWCONF_PIC/L_Hem', dict_eval['individual_lm_pic_dist_intersection_lowconf'][6], i)
            writer.add_scalar('LOWCONF_PIC/R_Hem', dict_eval['individual_lm_pic_dist_intersection_lowconf'][7], i)



            
            # DETECTED BUT NOT IN GT FOR EACH TYPE OF LANDMARK
            writer.add_scalar('PRED_N_GT/L_col', dict_eval['in_pred_not_gt'][0], i)
            writer.add_scalar('PRED_N_GT/R_col', dict_eval['in_pred_not_gt'][1], i)
            writer.add_scalar('PRED_N_GT/L_Sle', dict_eval['in_pred_not_gt'][2], i)
            writer.add_scalar('PRED_N_GT/R_Sle', dict_eval['in_pred_not_gt'][3], i)
            writer.add_scalar('PRED_N_GT/L_Wai', dict_eval['in_pred_not_gt'][4], i)
            writer.add_scalar('PRED_N_GT/R_Wai', dict_eval['in_pred_not_gt'][5], i)
            writer.add_scalar('PRED_N_GT/L_Hem', dict_eval['in_pred_not_gt'][6], i)
            writer.add_scalar('PRED_N_GT/R_Hem', dict_eval['in_pred_not_gt'][7], i)



            writer.add_scalar('GT_N_PRED/L_col', dict_eval['in_gt_not_pred'][0], i)
            writer.add_scalar('GT_N_PRED/R_col', dict_eval['in_gt_not_pred'][1], i)
            writer.add_scalar('GT_N_PRED/L_Sle', dict_eval['in_gt_not_pred'][2], i)
            writer.add_scalar('GT_N_PRED/R_Sle', dict_eval['in_gt_not_pred'][3], i)
            writer.add_scalar('GT_N_PRED/L_Wai', dict_eval['in_gt_not_pred'][4], i)
            writer.add_scalar('GT_N_PRED/R_Wai', dict_eval['in_gt_not_pred'][5], i)
            writer.add_scalar('GT_N_PRED/L_Hem', dict_eval['in_gt_not_pred'][6], i)
            writer.add_scalar('GT_N_PRED/R_Hem', dict_eval['in_gt_not_pred'][7], i)

            writer.add_scalar('PIC_N_PRED/L_col', dict_eval['in_pic_not_pred'][0], i)
            writer.add_scalar('PIC_N_PRED/R_col', dict_eval['in_pic_not_pred'][1], i)
            writer.add_scalar('PIC_N_PRED/L_Sle', dict_eval['in_pic_not_pred'][2], i)
            writer.add_scalar('PIC_N_PRED/R_Sle', dict_eval['in_pic_not_pred'][3], i)
            writer.add_scalar('PIC_N_PRED/L_Wai', dict_eval['in_pic_not_pred'][4], i)
            writer.add_scalar('PIC_N_PRED/R_Wai', dict_eval['in_pic_not_pred'][5], i)
            writer.add_scalar('PIC_N_PRED/L_Hem', dict_eval['in_pic_not_pred'][6], i)
            writer.add_scalar('PIC_N_PRED/R_Hem', dict_eval['in_pic_not_pred'][7], i)


            writer.add_scalar('PIC_N_GT/L_col', dict_eval['in_pic_not_gt'][0], i)
            writer.add_scalar('PIC_N_GT/R_col', dict_eval['in_pic_not_gt'][1], i)
            writer.add_scalar('PIC_N_GT/L_Sle', dict_eval['in_pic_not_gt'][2], i)
            writer.add_scalar('PIC_N_GT/R_Sle', dict_eval['in_pic_not_gt'][3], i)
            writer.add_scalar('PIC_N_GT/L_Wai', dict_eval['in_pic_not_gt'][4], i)
            writer.add_scalar('PIC_N_GT/R_Wai', dict_eval['in_pic_not_gt'][5], i)
            writer.add_scalar('PIC_N_GT/L_Hem', dict_eval['in_pic_not_gt'][6], i)
            writer.add_scalar('PIC_N_GT/R_Hem', dict_eval['in_pic_not_gt'][7], i)
            

            tensored_annotated_imgs = torch.tensor(np.array(annotated_imgs))
            print('all the information ', len(tensored_annotated_imgs.shape), '   ', tensored_annotated_imgs.shape)
            writer.add_images('images_'+str(i), tensored_annotated_imgs, i, dataformats = 'NHWC')


    writer.close()