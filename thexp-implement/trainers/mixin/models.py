from torch import nn

import torch

from thexp import Trainer
from thexp.contrib import EMA, ParamGrouper

import arch
from .. import GlobalParams


class ModelMixin(Trainer):

    def models(self, params: GlobalParams):
        raise NotImplementedError()

    def predict(self, xs):
        raise NotImplementedError()


class BaseModelMixin(ModelMixin):
    """base end-to-end model"""

    def predict(self, xs) -> torch.Tensor:
        with torch.no_grad():
            return self.ema_model(xs)

    def models(self, params: GlobalParams):
        if params.architecture == 'WRN':
            from arch.wideresnet import WideResNet
            self.model = WideResNet(depth=params.depth,
                                    widen_factor=params.widen_factor,
                                    with_fc=True,
                                    num_classes=params.n_classes)
        elif params.architecture == 'Resnet':
            from arch import resnet
            model_name = 'resnet{}'.format(params.depth)
            assert model_name in resnet.__dict__
            self.model = resnet.__dict__[model_name](num_classes=params.n_classes)
        else:
            assert False

        self.ema_model = EMA(self.model)

        grouper = ParamGrouper(self.model)
        noptim = params.optim.args.copy()
        noptim['weight_decay'] = 0
        param_groups = [
            grouper.create_param_group(grouper.kernel_params(with_norm=False), **params.optim.args),
            grouper.create_param_group(grouper.bias_params(with_norm=False), **noptim),
            grouper.create_param_group(grouper.norm_params(), **noptim),
        ]

        self.optim = params.optim.build(param_groups)
        self.to(self.device)
