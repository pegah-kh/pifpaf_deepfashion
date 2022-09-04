import openpifpaf
from .deepfashion_datamodule import Full_DeepFashionKP

def register():
    openpifpaf.DATAMODULES['full_right_vis_deepfashion'] = Full_DeepFashionKP