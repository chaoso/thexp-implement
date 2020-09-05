"""
reimplement of 《Learning to Reweight Examples for Robust Deep Learning》(L2R)，noisy label part
    https://arxiv.org/abs/1803.09050
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
from torch.nn import functional as F

from arch.meta import MetaSGD, MetaWideResNet
from trainers import NoisyParams
from trainers.mixin import *


class L2RNoisyLoss(losses.Loss):
    """
    先noisy 计算一遍loss，计算梯度，然后再clean计算一遍loss，计算梯度，然后梯度对梯度计算权重

    noisy的时候，为交叉熵求和reduction+zero weight
    clean的时候，为交叉熵求和reduction+1/batch weight ，等价于mean CE
        所有的损失同时再加上 L2 损失

    """

    def regularization_loss(self, model, l2_decay, meter: Meter, name: str):
        from thexp.contrib import ParamGrouper

        params = ParamGrouper(model).kernel_params(with_norm=False)
        cost = 0
        for p in params:
            cost = cost + (p ** 2).sum()

        meter[name] = cost * l2_decay
        return meter[name]

    def weighted_ce_loss(self, logits, labels, weights, meter: Meter, name: str):
        loss_ = F.cross_entropy(logits, labels, reduction='none')

        loss = torch.sum(loss_ * weights)
        meter[name] = loss
        return loss


class L2RTrainer(callbacks.BaseCBMixin,
                 models.BaseModelMixin,
                 acc.ClassifyAccMixin,
                 losses.CELoss, L2RNoisyLoss,
                 Trainer):

    def datasets(self, params: NoisyParams):
        from data.dataxy import cifar10
        from thexp.contrib.data import splits
        from data.constant import norm_val
        from thexp import DataBundler, DatasetBuilder
        from data.transforms import Weak, ToNormTensor

        test_x, test_y = cifar10(False)
        train_x, train_y = cifar10(True)

        train_ids, val_ids = splits.train_val_split(train_y, val_size=5000)

        train_x, val_x = train_x[train_ids], train_x[val_ids]
        train_y, val_y = train_y[train_ids], train_y[val_ids]

        if params.noisy_type == 'asymmetric':
            from data.noisy import asymmetric_noisy
            noisy_y = asymmetric_noisy(train_y, params.noisy_ratio, n_classes=params.n_classes)

        elif params.noisy_type == 'symmetric':
            from data.noisy import symmetric_noisy
            noisy_y = symmetric_noisy(train_y, params.noisy_ratio, n_classes=params.n_classes)

        else:
            assert False

        self.logger.info('noisy acc = {}'.format((train_y == noisy_y).mean()))
        self.rnd.shuffle()

        mean, std = norm_val.get('cifar10', [None, None])
        toTensor = ToNormTensor(mean, std)
        weak = Weak(mean, std)

        train_set = (
            DatasetBuilder(train_x, train_y)
                .add_labels(noisy_y, 'noisy_y')
                .add_x(transform=weak)
                .add_y()
                .add_y(source='noisy_y')
        )
        self.train_set = train_set
        train_dataloader = train_set.DataLoader(batch_size=params.batch_size, num_workers=params.num_workers,
                                                shuffle=True)

        val_dataloader = (
            DatasetBuilder(val_x, val_y)
                .add_x(transform=toTensor).add_y()
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True)
        )

        test_dataloader = (
            DatasetBuilder(test_x, test_y)
                .add_x(transform=toTensor).add_y()
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True)
        )

        self.regist_databundler(train=DataBundler().add(train_dataloader).cycle(val_dataloader).zip_mode(),
                                eval=val_dataloader,
                                test=test_dataloader)
        self.to(self.device)

    def create_metanet(self):
        metanet = MetaWideResNet(num_classes=10).to(self.device)
        metanet.load_state_dict(self.model.state_dict())

        meta_sgd = MetaSGD(metanet, **self.params.meta_optim)
        return metanet, meta_sgd

    def train_batch(self, eidx, idx, global_step, batch_data, params: NoisyParams, device: torch.device):
        meter = Meter()
        # 注意，clean_label 为训练集的真实label，只用于计算准确率，不用于训练过程
        (images, clean_labels, labels), (val_images, val_labels) = batch_data

        metanet, meta_sgd = self.create_metanet()
        y_f_hat = metanet(images)
        cost = F.cross_entropy(y_f_hat, labels, reduction='none')

        eps = torch.zeros_like(labels, device=device, requires_grad=True, dtype=torch.float32)
        l_f_meta = self.weighted_ce_loss(y_f_hat, labels, eps, meter, 'a_ce') + \
                   self.regularization_loss(metanet, params.l2_decay, meter, 'a_l2')  # meta noisy ce

        metanet.zero_grad()

        grads = torch.autograd.grad(l_f_meta, (metanet.params()), create_graph=True, retain_graph=True)

        # metanet.update_params(params.optim.lr, grads=grads)

        y_g_hat = metanet(val_images)
        eps_b = torch.zeros_like(labels,
                                 device=device,
                                 requires_grad=True,
                                 dtype=torch.float32) + (1 / params.batch_size)

        v_meta_loss = self.weighted_ce_loss(y_g_hat, val_labels, eps_b, meter, 'b_ce') + \
                      self.regularization_loss(metanet, params.l2_decay, meter, 'b_l2')  # meta clean ce + meta clean l2

        grads_b = torch.autograd.grad(v_meta_loss, metanet.params(), create_graph=True, retain_graph=True)
        # grad_eps = torch.autograd.grad(v_meta_loss, eps, only_inputs=True)[0]
        grad_eps = torch.autograd.grad(grads, eps, grad_outputs=grads_b)[0]

        w_tilde = torch.clamp_min(grad_eps, min=0)
        norm_c = torch.sum(w_tilde)

        if norm_c != 0:
            w = w_tilde / norm_c
        else:
            w = w_tilde

        y_f_hat = self.model(images)  # type:torch.Tensor

        # l_f = F.cross_entropy(y_f_hat, clean_labels)
        cost = F.cross_entropy(y_f_hat, labels, reduction='none')
        l_f = torch.sum(cost * w.detach())

        self.optim.zero_grad()
        l_f.backward()
        self.optim.step()

        meter.all_loss = l_f
        # meter.meta_l = v_meta_loss
        meter.zero_r = (w == 0).float().mean()
        meter.acc = (y_f_hat.argmax(dim=-1) == clean_labels).float().mean()
        meter.noisy_acc = (y_f_hat.argmax(dim=-1) == labels).float().mean()
        meter.wacc = ((y_f_hat.argmax(dim=-1) == clean_labels).float() * w).mean()
        meter.val_acc = (y_g_hat.argmax(dim=-1) == val_labels).float().mean()
        meter.percent(meter.acc_)
        meter.percent(meter.wacc_)
        meter.percent(meter.noisy_acc_)
        meter.percent(meter.val_acc_)

        return meter

    def test_eval_logic(self, dataloader, param: NoisyParams):
        meter = AvgMeter()
        for itr, (images, labels) in enumerate(dataloader):
            output = self.model(images).squeeze()
            predicted = (torch.sigmoid(output) > 0.5).int()
            meter.acc = (predicted.int() == labels.int()).float().mean().detach()

        return meter


if __name__ == '__main__':
    params = NoisyParams()
    params.ema = False  # l2r have no ema for model
    params.epoch = 120
    params.batch_size = 5
    params.device = 'cpu'
    params.optim.args.lr = 0.1
    params.meta_optim = {
        'lr': 0.1,
        'momentum': 0.9,
    }
    params.from_args()
    trainer = L2RTrainer(params)

    trainer.train()
