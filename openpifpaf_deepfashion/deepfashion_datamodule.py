import argparse
import os

import torch

import openpifpaf
from openpifpaf.plugins.coco.dataset import CocoDataset
from openpifpaf import metric


try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None

from .constants import (
    DEEPFASHION_SKELETON,
    DEEPFASHION_KEYPOINTS,
    DEEPFASHION_POSE,
    HFLIP,
    DEEPFASHION_SIGMAS,
    DEEPFASHION_SCORE_WEIGHTS,
    DEEPFASHION_CATEGORIES
)

try:
    import pycocotools.coco
    # monkey patch for Python 3 compat
    pycocotools.coco.unicode = str
except ImportError:
    pass



class Full_DeepFashionKP(openpifpaf.datasets.DataModule, openpifpaf.Configurable):
    # TO BE CHANGED BASED ON WHERE YOU KEEP YOUR DATASET AND JSON FILES
    train_annotations = '../../pifpaf_deepfahsion/full_right_vis/train_annotations_MSCOCO_style.json'
    val_annotations = '../../pifpaf_deepfahsion/full_right_vis/valid_annotations_MSCOCO_style.json'

    eval_annotations = val_annotations
    train_image_dir = '../../pifpaf_deepfahsion/full_right_vis/img_train'
    val_image_dir = '../../pifpaf_deepfahsion/full_right_vis/img_valid'
    eval_image_dir = val_image_dir

    square_edge = 512
    with_dense = False
    extended_scale = False
    orientation_invariant = 0.0
    blur = 0.0
    augmentation = True
    rescale_images = 1.0
    upsample_stride = 1
    min_kp_anns = 1
    bmin = 0.1

    eval_annotation_filter = True
    eval_long_edge = 512
    eval_orientation_invariant = 0.0
    eval_extended_scale = False

    skeleton = DEEPFASHION_SKELETON

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        cif = openpifpaf.headmeta.Cif('cif', 'deepfashion',
                                      keypoints=DEEPFASHION_KEYPOINTS,
                                      sigmas=DEEPFASHION_SIGMAS,
                                      pose=DEEPFASHION_POSE,
                                      draw_skeleton=self.skeleton,
                                      score_weights=DEEPFASHION_SCORE_WEIGHTS)
        caf = openpifpaf.headmeta.Caf('caf', 'deepfashion',
                                      keypoints=DEEPFASHION_KEYPOINTS,
                                      sigmas=DEEPFASHION_SIGMAS,
                                      pose=DEEPFASHION_POSE,
                                      skeleton=self.skeleton)

        cif.upsample_stride = self.upsample_stride
        caf.upsample_stride = self.upsample_stride
        self.head_metas = [cif, caf] if self.with_dense else [cif, caf]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module deepfashionkp')

        group.add_argument('--deepfashion-train-annotations', default=cls.train_annotations,
                           help='train annotations')
        group.add_argument('--deepfashion-val-annotations', default=cls.val_annotations,
                           help='val annotations')
        group.add_argument('--deepfashion-train-image-dir', default=cls.train_image_dir,
                           help='train image dir')
        group.add_argument('--deepfashion-val-image-dir', default=cls.val_image_dir,
                           help='val image dir')

        group.add_argument('--deepfashion-square-edge',
                           default=cls.square_edge, type=int,
                           help='square edge of input images')
        assert not cls.with_dense
        group.add_argument('--deepfashion-with-dense',
                           default=False, action='store_true',
                           help='train with dense connections')
        assert not cls.extended_scale
        group.add_argument('--deepfashion-extended-scale',
                           default=False, action='store_true',
                           help='augment with an extended scale range')
        group.add_argument('--deepfashion-orientation-invariant',
                           default=cls.orientation_invariant, type=float,
                           help='augment with random orientations')
        group.add_argument('--deepfashion-blur',
                           default=cls.blur, type=float,
                           help='augment with blur')
        assert cls.augmentation
        group.add_argument('--deepfashion-no-augmentation',
                           dest='deepfashion_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--deepfashion-rescale-images',
                           default=cls.rescale_images, type=float,
                           help='overall rescale factor for images')
        group.add_argument('--deepfashion-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')
        group.add_argument('--deepfashion-min-kp-anns',
                           default=cls.min_kp_anns, type=int,
                           help='filter images with fewer keypoint annotations')
        group.add_argument('--deepfashion-bmin',
                           default=cls.bmin, type=float,
                           help='bmin')

        assert cls.eval_annotation_filter
        group.add_argument('--deepfashion-no-eval-annotation-filter',
                           dest='deepfashion_eval_annotation_filter',
                           default=True, action='store_false')
        group.add_argument('--deepfashion-eval-long-edge', default=cls.eval_long_edge, type=int,
                           help='set to zero to deactivate rescaling')
        assert not cls.eval_extended_scale
        group.add_argument('--deepfashion-eval-extended-scale', default=False, action='store_true')
        group.add_argument('--deepfashion-eval-orientation-invariant',
                           default=cls.eval_orientation_invariant, type=float)

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # deepfashion specific
        cls.train_annotations = args.deepfashion_train_annotations
        cls.val_annotations = args.deepfashion_val_annotations
        cls.train_image_dir = args.deepfashion_train_image_dir
        cls.val_image_dir = args.deepfashion_val_image_dir

        cls.square_edge = args.deepfashion_square_edge
        cls.with_dense = args.deepfashion_with_dense
        cls.extended_scale = args.deepfashion_extended_scale
        cls.orientation_invariant = args.deepfashion_orientation_invariant
        cls.blur = args.deepfashion_blur
        cls.augmentation = args.deepfashion_augmentation
        cls.rescale_images = args.deepfashion_rescale_images
        cls.upsample_stride = args.deepfashion_upsample
        cls.min_kp_anns = args.deepfashion_min_kp_anns
        cls.bmin = args.deepfashion_bmin

        # evaluation
        cls.eval_annotation_filter = args.deepfashion_eval_annotation_filter
        cls.eval_long_edge = args.deepfashion_eval_long_edge
        cls.eval_orientation_invariant = args.deepfashion_eval_orientation_invariant
        cls.eval_extended_scale = args.deepfashion_eval_extended_scale

    def _preprocess(self):
        encoders = [openpifpaf.encoder.Cif(self.head_metas[0], bmin=self.bmin),
                    openpifpaf.encoder.Caf(self.head_metas[1], bmin=self.bmin)]
        if len(self.head_metas) > 2:
            encoders.append(openpifpaf.encoder.Caf(self.head_metas[2], bmin=self.bmin))

        if not self.augmentation:
            return openpifpaf.transforms.Compose([
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.RescaleAbsolute(self.square_edge),
                openpifpaf.transforms.CenterPad(self.square_edge),
                openpifpaf.transforms.EVAL_TRANSFORM,
                openpifpaf.transforms.Encoders(encoders),
            ])

        if self.extended_scale:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.25 * self.rescale_images,
                             2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))
        else:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.4 * self.rescale_images,
                             2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))

        return openpifpaf.transforms.Compose([
            openpifpaf.transforms.NormalizeAnnotations(),
            openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.HFlip(DEEPFASHION_KEYPOINTS, HFLIP), 0.5),
            rescale_t,
            openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.Blur(), self.blur),
            openpifpaf.transforms.RandomChoice(
                [openpifpaf.transforms.RotateBy90(),
                 openpifpaf.transforms.RotateUniform(30.0)],
                [self.orientation_invariant, 0.4],
            ),
            openpifpaf.transforms.Crop(self.square_edge, use_area_of_interest=True),
            openpifpaf.transforms.CenterPad(self.square_edge),
            openpifpaf.transforms.TRAIN_TRANSFORM,
            openpifpaf.transforms.Encoders(encoders),
        ])

    def train_loader(self):
        train_data = CocoDataset(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[1],
        )
        directory = os.getcwd()
        print(directory)
        print('************',  train_data.__len__(), '***********')
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug and self.augmentation,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def val_loader(self):
        val_data = CocoDataset(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[1],
        )
        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=not self.debug and self.augmentation,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    @classmethod
    def common_eval_preprocess(cls):
        rescale_t = None
        if cls.eval_extended_scale:
            assert cls.eval_long_edge
            rescale_t = [
                openpifpaf.transforms.DeterministicEqualChoice([
                    openpifpaf.transforms.RescaleAbsolute(cls.eval_long_edge),
                    openpifpaf.transforms.RescaleAbsolute((cls.eval_long_edge - 1) // 2 + 1),
                ], salt=1)
            ]
        elif cls.eval_long_edge:
            rescale_t = openpifpaf.transforms.RescaleAbsolute(cls.eval_long_edge)

        if cls.batch_size == 1:
            padding_t = openpifpaf.transforms.CenterPadTight(16)
        else:
            assert cls.eval_long_edge
            padding_t = openpifpaf.transforms.CenterPad(cls.eval_long_edge)

        orientation_t = None
        if cls.eval_orientation_invariant:
            orientation_t = openpifpaf.transforms.DeterministicEqualChoice([
                None,
                openpifpaf.transforms.RotateBy90(fixed_angle=90),
                openpifpaf.transforms.RotateBy90(fixed_angle=180),
                openpifpaf.transforms.RotateBy90(fixed_angle=270),
            ], salt=3)

        return [
            openpifpaf.transforms.NormalizeAnnotations(),
            rescale_t,
            padding_t,
            orientation_t,
        ]