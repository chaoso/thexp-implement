"""
reimplement of 《FixMatch: Simplifying Semi-Supervised Learning with Consistency and Conﬁdence》
    https://arxiv.org/abs/2001.07685

a small change of fixmatch: create an exponential moving average of the pseudo target of unlabeled data.
"""
if __name__ == '__main__':
    import os
    import sys

    chdir = os.path.dirname(os.path.abspath(__file__))
    chdir = os.path.dirname(chdir)
    chdir = os.path.dirname(chdir)
    sys.path.append(chdir)

import torch

from thexp import Trainer, Meter
from trainers import SemiSupervisedParams

from trainers.mixin import *


class FixMatchParams(SemiSupervisedParams):

    def __init__(self):
        super().__init__()
        self.epoch = 16912  # 62 iter / epoch,
        self.batch_size = 64
        self.optim = self.create_optim('SGD',
                                       lr=0.03,
                                       weight_decay=0.0005,
                                       momentum=0.9,
                                       nesterov=True)
        self.pred_thresh = 0.95

        self.lambda_u = 1
        self.targets_ema = 0.25
        self.uratio = 7
        self.bind('datasets', 'cifar100', 'optim.weight_decay', 0.001)


class FixMatchV2Trainer(callbacks.BaseCBMixin,
                        datasets.FixMatchDatasetMixin,
                        models.BaseModelMixin,
                        acc.ClassifyAccMixin,
                        losses.CELoss, losses.FixMatchLoss,
                        Trainer):

    def initial(self):
        super().initial()
        # first, create a tensor
        self.targets_mem = torch.zeros(50000, params.n_classes, dtype=torch.float).to(self.device)

    def train_batch(self, eidx, idx, global_step, batch_data, params: FixMatchParams, device: torch.device):
        super().train_batch(eidx, idx, global_step, batch_data, params, device)
        meter = Meter()

        sup, unsup = batch_data
        xs, ys = sup
        ids, un_xs, un_axs, un_ys = unsup

        logits_list = self.to_logits(torch.cat([xs, un_xs, un_axs])).split_with_sizes(
            [xs.shape[0], un_xs.shape[0], un_xs.shape[0]])
        logits, un_w_logits, un_s_logits = logits_list  # type:torch.Tensor

        raw_targets = torch.softmax(un_w_logits, dim=-1)

        # second, update the tensor by moving average and use it as the final targets of the unlabeled data
        with torch.no_grad():
            targets = self.targets_mem[ids] * params.targets_ema + raw_targets * (1 - params.targets_ema)
            self.targets_mem[ids] = targets

        max_probs, un_pseudo_labels = torch.max(targets, dim=-1)
        mask = (max_probs > params.pred_thresh).float()

        meter.all_loss = meter.all_loss + self.loss_ce_(logits, ys, meter, name='Lx')
        meter.all_loss = meter.all_loss + self.loss_ce_with_masked_(un_s_logits, un_pseudo_labels, mask,
                                                                    meter, name='Lu') * params.lambda_u

        self.optim.zero_grad()
        meter.all_loss.backward()
        self.optim.step()

        meter.masked = mask.float().mean()
        self.acc_precise_(logits.argmax(dim=1), ys, meter)
        self.acc_precise_(un_w_logits.argmax(dim=1), un_ys, meter, name='uwacc')
        self.acc_precise_(un_s_logits.argmax(dim=1), un_ys, meter, name='usacc')

        return meter

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)


if __name__ == '__main__':
    params = FixMatchParams()
    params.device = 'cuda:3'
    params.from_args()
    trainer = FixMatchV2Trainer(params)
    trainer.train()
