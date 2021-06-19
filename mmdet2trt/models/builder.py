import logging

import mmdet  # noqa: F401,F403
from mmdet import models  # noqa: F401,F403

import mmcv.ops  # noqa: F401,F403

WRAPER_DICT = {}


def register_wraper(module_name):
    try:
        mmdet_module = eval(module_name)

        def register_func(wrap_cls):
            if mmdet_module in WRAPER_DICT:
                logging.warning(
                    '{} is already registed.'.format(mmdet_module) +
                    ' new wraper {} will cover current wraper {}.'.format(
                        wrap_cls, WRAPER_DICT[mmdet_module]))
            WRAPER_DICT[mmdet_module] = wrap_cls
            return wrap_cls

        return register_func

    except Exception:
        logging.warn('module {} not exist.'.format(module_name))

        def register_func(wrap_cls):
            return wrap_cls

        return register_func


def build_wraper(module, default_wraper=None, **kwargs):
    model_type = module.__class__

    wrap_model = None
    if model_type in WRAPER_DICT:
        logging.debug('find module type:{}'.format(str(model_type)))
        wrap_model = WRAPER_DICT[model_type](module, **kwargs)
    else:
        logging.warning(
            "can't find wrap module for type:{}, use {} instead.".format(
                str(model_type), default_wraper))
        if default_wraper is not None:
            wrap_model = default_wraper(module)

    return wrap_model
