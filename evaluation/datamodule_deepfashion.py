from ast import arg
import torch
import openpifpaf
import transforms
from openpifpaf.plugins.coco.dataset import CocoDataset

from constants import (
    DEEPFASHION_SKELETON,
    DEEPFASHION_KEYPOINTS,
    DEEPFASHION_POSE,
    HFLIP,
    DEEPFASHION_SIGMAS,
    DEEPFASHION_SCORE_WEIGHTS,
    DEEPFASHION_CATEGORIES
)




def _eval_preprocess():
    cif = openpifpaf.headmeta.Cif('cif', 'deepfashion',
                                    keypoints=DEEPFASHION_KEYPOINTS,
                                    sigmas=DEEPFASHION_SIGMAS,
                                    pose=DEEPFASHION_POSE,
                                    draw_skeleton=DEEPFASHION_SKELETON,
                                    score_weights=DEEPFASHION_SCORE_WEIGHTS)
    caf = openpifpaf.headmeta.Caf('caf', 'deepfashion',
                                    keypoints=DEEPFASHION_KEYPOINTS,
                                    sigmas=DEEPFASHION_SIGMAS,
                                    pose=DEEPFASHION_POSE,
                                    skeleton=DEEPFASHION_SKELETON)

        
    head_metas = [cif, caf]
    return openpifpaf.transforms.Compose([
        transforms.BBoxCrop(),
        transforms.CheckLandmarks(),
        openpifpaf.transforms.NormalizeAnnotations(),
        openpifpaf.transforms.RescaleAbsolute(520), # use different edges for category classification and landmark localization
        openpifpaf.transforms.CenterPad(520),
        openpifpaf.transforms.EVAL_TRANSFORM,
    ])



def eval_loader(args):
    eval_image_dir = args.eval_image_dir
    eval_annotations = args.eval_annotations
    batch_size = args.batch_size


    eval_data = CocoDataset(
        image_dir=eval_image_dir,
        ann_file=eval_annotations,
        preprocess=_eval_preprocess(),
        annotation_filter=True,
        min_kp_anns=1,
    )
    return torch.utils.data.DataLoader(
        eval_data, batch_size=batch_size, shuffle=False, drop_last=False,
        collate_fn=openpifpaf.datasets.collate_images_anns_meta)
