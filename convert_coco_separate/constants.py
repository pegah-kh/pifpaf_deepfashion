# question: what to do with POSE?!?! no idea of the scale used in other codes


import numpy as np


DEEPFASHION_SKELETON = [
    [1, 2], # left and right collar connection
    [1, 3],
    [1, 5], # left hem and left sleeve connection
    [5, 6], # left and right hem connection I AM NOT SURE IF THIS SHOULD BE IN THE SKELETON
    [6, 2], # right sleeve and right collar connection
    [4, 2],
    [5, 7],
    [7, 8],
    [6, 8]
]



DEEPFASHION_KEYPOINTS = [
    "left_collar", # 1
    "right_collar", # 2
    "left_sleeve", # 3
    "right_sleeve", # 4
    "left_waistline", # 5
    "right_waistline", # 6
    "left_hem",
    "right_hem"
]



DEEPFASHION_POSE = np.array([ # these are just random numbers....
    # from coco plugin....
    [-1.4, 8.0, 2.0],  # 'left_collar' # 1
    [1.4, 8.0, 2.0],  # 'right_collar' # 2
    [-1.75, 4.0, 2.0],  # 'left_sleeve' # 3
    [1.75, 4.2, 2.0],  # 'right_sleeve' # 4
    [-1.26, 4.0, 2.0],  # 'left_waistline' # 5
    [1.26, 4.0, 2.0],  # 'right_waistline' # 6
    [-1.4, 0.0, 2.0],  # 'left_hem' # 7
    [1.4, 0.1, 2.0],  # 'right_hem' # 8
])



HFLIP = {
    'left_waistline': 'right_waistline',
    'right_hem': 'left_hem',
    'left_sleeve': 'right_sleeve',
    'right_collar': 'left_collar',
}


DEEPFASHION_SIGMAS = [
    0.079,  # right_collar
    0.079,  # left_collar
    0.062,  # left_sleeve
    0.062,  # right_sleeve
    0.107,  # left_waistline
    0.107,  # right_waistline
    0.089,  # right_hem
    0.089,  # left_hem
]


 
DEEPFASHION_SCORE_WEIGHTS = [3.0] * 4 + [1.0] * (len(DEEPFASHION_KEYPOINTS) - 4)


DEEPFASHION_CATEGORIES = "clothing"


def draw_skeletons(pose):
    import openpifpaf  # pylint: disable=import-outside-toplevel
    openpifpaf.show.KeypointPainter.show_joint_scales = True
    keypoint_painter = openpifpaf.show.KeypointPainter()

    scale = np.sqrt(
        (np.max(pose[:, 0]) - np.min(pose[:, 0]))
        * (np.max(pose[:, 1]) - np.min(pose[:, 1]))
    )

    ann = openpifpaf.Annotation(keypoints=DEEPFASHION_KEYPOINTS,
                                skeleton=DEEPFASHION_SKELETON,
                                score_weights=DEEPFASHION_SCORE_WEIGHTS)
    ann.set(pose, np.array(DEEPFASHION_SIGMAS) * scale)
    with openpifpaf.show.Canvas.annotation(
            ann, filename='docs/skeleton_deepfashion.png') as ax:
        keypoint_painter.annotation(ax, ann)


def print_associations():
    for j1, j2 in DEEPFASHION_SKELETON:
        print(DEEPFASHION_KEYPOINTS[j1 - 1], '-', DEEPFASHION_KEYPOINTS[j2 - 1])


if __name__ == '__main__':
    print_associations()
    draw_skeletons(DEEPFASHION_POSE)