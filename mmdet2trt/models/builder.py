import logging
import mmdet
from mmdet import models
import mmcv.ops

WARPER_DICT = {}

def register_warper(module_name):
    try:
        mmdet_module = eval(module_name)
        def register_func(warp_cls):
            if mmdet_module in WARPER_DICT:
                logging.warning("{} is already registed. new warper {} will cover current warper {}.".format(mmdet_module, warp_cls, WARPER_DICT[mmdet_module]))
            WARPER_DICT[mmdet_module] = warp_cls
            return warp_cls
        return register_func

    except:
        logging.warn("module {} not exist.".format(module_name))
        def register_func(warp_cls):
            return warp_cls
        
        return register_func


def build_warper(module, default_warper=None):
    model_type = module.__class__

    warp_model = None
    if model_type in WARPER_DICT:
        logging.debug("find module type:{}".format(str(model_type)))
        warp_model = WARPER_DICT[model_type](module)
    else:
        logging.warning("can't find warp module for type:{}, use {} instead.".format(str(model_type), default_warper))
        if default_warper is not None:
            warp_model = default_warper(module)

    return warp_model
    
