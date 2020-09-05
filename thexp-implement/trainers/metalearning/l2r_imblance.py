"""
reimplement of 《Learning to Reweight Examples for Robust Deep Learning》(L2R)，imblance part
    https://arxiv.org/abs/1803.09050

train 25 epoch is enough to see the result(can achieve 93%-95% accuracy), where basic methods can't train anything(about 50%)
"""
if __name__ == '__main__':
    import os
    import sys

    chdir = os.path.dirname(os.path.abspath(__file__))
    chdir = os.path.dirname(chdir)
    chdir = os.path.dirname(chdir)
    sys.path.append(chdir)

import torch

from thexp import Trainer, Meter, AvgMeter
from trainers import MnistImblanceParams
from trainers.mixin import *
from arch.meta import MetaLeNet
from torch.nn import functional as F


class L2RTrainer(datasets.MnistImblanceDataset,
                 callbacks.BaseCBMixin,
                 acc.ClassifyAccMixin,
                 losses.CELoss,
                 Trainer):

    def models(self, params: MnistImblanceParams):
        from arch.lenet import LeNet
        from thexp.contrib import ParamGrouper

        self.model = LeNet(1)  # type:nn.Module

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

    def train_batch(self, eidx, idx, global_step, batch_data, params: MnistImblanceParams, device: torch.device):
        meter = Meter()

        (images, labels), (val_images, val_labels) = batch_data  # type:torch.Tensor

        metanet = MetaLeNet(1).to(device)
        metanet.load_state_dict(self.model.state_dict())
        y_f_hat = metanet(images).squeeze()
        cost = F.binary_cross_entropy_with_logits(y_f_hat, labels, reduction='none')
        eps = torch.zeros_like(labels, device=device, requires_grad=True)
        l_f_meta = torch.sum(cost * eps)
        metanet.zero_grad()

        grads = torch.autograd.grad(l_f_meta, (metanet.params()), create_graph=True)
        metanet.update_params(params.optim.args.lr, grads=grads)

        y_g_hat = metanet(val_images).squeeze()
        v_meta_loss = F.binary_cross_entropy_with_logits(y_g_hat, val_labels)
        grad_eps = torch.autograd.grad(v_meta_loss, eps, only_inputs=True)[0]
        w_tilde = torch.clamp(-grad_eps, min=0)
        norm_c = torch.sum(w_tilde)

        if norm_c != 0:
            w = w_tilde / norm_c
        else:
            w = w_tilde

        y_f_hat = self.model(images).squeeze()
        cost = F.binary_cross_entropy_with_logits(y_f_hat, labels, reduction='none')
        l_f = torch.sum(cost * w.detach())

        self.optim.zero_grad()
        l_f.backward()
        self.optim.step()

        meter.l_f = l_f
        meter.meta_l = v_meta_loss
        if (labels == 0).sum() > 0:
            meter.grad_0 = grad_eps[labels == 0].mean() * 1e5
            meter.grad_0_max = grad_eps[labels == 0].max() * 1e5
            meter.grad_0_min = grad_eps[labels == 0].min() * 1e5
        if (labels == 1).sum() > 0:
            meter.grad_1 = grad_eps[labels == 1].mean() * 1e5
            meter.grad_1_max = grad_eps[labels == 1].max() * 1e5
            meter.grad_1_min = grad_eps[labels == 1].min() * 1e5

        return meter

    def test_eval_logic(self, dataloader, param: MnistImblanceParams):
        meter = AvgMeter()
        for itr, (images, labels) in enumerate(dataloader):
            output = self.model(images).squeeze()
            predicted = (torch.sigmoid(output) > 0.5).int()
            meter.acc = (predicted.int() == labels.int()).float().mean().detach()

        return meter


if __name__ == '__main__':
    params = MnistImblanceParams()
    params.device = 'cpu'
    params.from_args()
    trainer = L2RTrainer(params)

    trainer.train()
