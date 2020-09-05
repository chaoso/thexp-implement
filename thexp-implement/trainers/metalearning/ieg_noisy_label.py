"""

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
from trainers import NoisyParams
from trainers.mixin import *
from arch.meta import MetaLeNet
from torch.nn import functional as F
from arch.meta import MetaSGD, MetaWideResNet


class IEGTrainer(callbacks.BaseCBMixin,
                 models.BaseModelMixin,
                 acc.ClassifyAccMixin,
                 losses.CELoss, losses.IEGLossMixin, losses.MixMatchMixin,
                 Trainer):

    def datasets(self, params: NoisyParams):
        from data.dataxy import cifar10
        from thexp.contrib.data import splits
        from data.constant import norm_val
        from thexp import DataBundler, DatasetBuilder
        from data.transforms import Weak, ToNormTensor, Strong

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
        strong = Strong(mean, std)

        train_set = (
            DatasetBuilder(train_x, train_y)
                .add_labels(noisy_y, 'noisy_y')
                .add_x(transform=weak)
                .add_x(transform=strong)
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

    def semi_mixmatch_loss(self,
                           sup_imgs: torch.Tensor,
                           sup_labels: torch.Tensor,
                           *un_imgs: torch.Tensor,
                           un_logits, un_aug_logits,
                           meter: Meter):
        sup_targets = tricks.onehot(sup_labels, self.params.n_classes)

        un_targets = self.label_guesses(un_logits, un_aug_logits)

        if self.params.sharpen_T != 0:
            un_targets = self.sharpen(un_targets, self.params.sharpen_T)

        mixed_input, mixed_target = self.mixmatch_up(sup_imgs, un_imgs, sup_targets, un_targets)

        # 将有标签数据和无标签数据分开
        sup_mixed_target, unsup_mixed_target = mixed_target.split_with_sizes(
            [sup_imgs.shape[0], mixed_input.shape[0] - sup_imgs.shape[0]])

        sup_mixed_logits, unsup_mixed_logits = self.to_logits(mixed_input).split_with_sizes(
            [sup_imgs.shape[0], mixed_input.shape[0] - sup_imgs.shape[0]])

        # 计算最终的损失

        meter.Lall = meter.Lall + self.loss_ce_with_targets_(sup_mixed_logits, sup_mixed_target,
                                                             meter=meter)  # + all_loss
        self.loss_ce_with_targets_(unsup_mixed_logits,
                                   unsup_mixed_target,
                                   w_ce=params.ce_factor,
                                   meter=meter)  # + all_loss

        return un_targets

    def train_batch(self, eidx, idx, global_step, batch_data, params: NoisyParams, device: torch.device):
        meter = Meter()
        # 注意，clean_label 为训练集的真实label，只用于计算准确率，不用于训练过程
        (images, aug_image, clean_labels, labels), (val_images, val_labels) = batch_data

        noisy_logits = self.to_logits(images)

        # other losses TODO
        # 三个损失，MixMatch的两个+一个KL散度损失

        logits = self.to_logits(images)
        aug_logits = self.to_logits(aug_image)

        guess_targets = self.semi_mixmatch_loss(
            val_images, val_labels,
            images, aug_image,
            un_logits=logits, un_aug_logits=aug_logits,
            meter=meter)

        meter.Lall = meter.Lall + self.loss_kl_ieg_(logits, aug_logits, params.kl_factor, meter)  # + all_loss

        weight_1, eps_1, noisy_targets = self.meta_optimize_(images, labels, guess_targets,
                                                             val_images, val_labels, meter)

        mixed_targets = eps_1 * noisy_targets + (1 - eps_1) * guess_targets
        net_loss1 = self.loss_softmax_cross_entropy_with_targets_(mixed_targets, logits,
                                                                  meter=meter, name='net_loss')

        init_mixed_labels = 0.9 * noisy_targets + 0.1 * guess_targets
        net_loss2 = self.weighted_loss_(noisy_logits, init_mixed_labels,
                                        weight_1,
                                        meter=meter, name='init_net_loss')

        meter.Lall = meter.Lall + (net_loss1 + net_loss2) / 2
        meter.Lall = meter.Lall + self.loss_regularization_(
            self.get_model(),
            params.l2_decay,
            meter=meter, name='l2_loss')

    def test_eval_logic(self, dataloader, params: NoisyParams):
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
    trainer = IEGTrainer(params)

    trainer.train()
