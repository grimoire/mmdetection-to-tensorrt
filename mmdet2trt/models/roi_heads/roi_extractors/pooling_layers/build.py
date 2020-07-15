from .roi_align_extractor import *
import logging

POOLING_DICT={
    "RoIAlign": RoiAlignExtractor
}

def build_roi_extractor(pooling_name, module):
    
    if pooling_name in POOLING_DICT:
        return POOLING_DICT[pooling_name](module)
    else:
        logging.warn("pooling type:{} not exist".format(pooling_name))
        return None