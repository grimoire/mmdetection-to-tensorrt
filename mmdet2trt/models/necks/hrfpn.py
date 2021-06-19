import torch.nn as nn
import torch.nn.functional as F
from mmdet2trt.models.builder import register_wraper


def pooling_wrap(pooling):
    if pooling.__name__.startswith('avg'):

        def pool(*args, **kwargs):
            return F.avg_pool2d(*args, **kwargs)

        return pool
    elif pooling.__name__.startswith('max'):

        def pool(*args, **kwargs):
            return F.max_pool2d(*args, **kwargs)

        return pool
    return None


@register_wraper('mmdet.models.necks.HRFPN')
class HRFPNWraper(nn.Module):

    def __init__(self, module):
        super(HRFPNWraper, self).__init__()
        self.module = module
        self.module.pooling = pooling_wrap(self.module.pooling)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
