from typing import Tuple

from thexp.nest.trainer.losses import *
from thexp.calculate.tensor import onehot
import numpy as np


class MixMatchMixin(Loss):

    def sharpen(self, x: torch.Tensor, T=0.5):
        """让概率分布变的更 sharp，即倾向于 onehot"""
        temp = x ** (1 / T)
        return temp / temp.sum(dim=1, keepdims=True)

    def label_guesses(self, *logits):
        """根据K次增广猜测"""
        k = len(logits)
        with torch.no_grad():
            un_logits = torch.cat(logits)  # type:torch.Tensor
            targets_u = torch.softmax(un_logits, dim=1) \
                            .view(k, -1, un_logits.shape[-1]) \
                            .sum(dim=0) / k
            targets_u = targets_u.detach()
            return targets_u

    def mixup(self,
              imgs: torch.Tensor, targets: torch.Tensor,
              beta=0.75):
        """
        普通的mixup操作
        """
        idx = torch.randperm(imgs.size(0))
        input_a, input_b = imgs, imgs[idx]
        target_a, target_b = targets, targets[idx]

        l = np.random.beta(beta, beta)
        l = max(l, 1 - l)
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        return mixed_input, mixed_target

    def mixmatch_up(self,
                    sup_imgs: torch.Tensor, un_sup_imgs: Tuple[torch.Tensor],
                    sup_targets: torch.Tensor, un_targets: torch.Tensor):
        """
        使用过MixMatch的方法对有标签和无标签数据进行mixup混合

        注意其中 un_sup_imgs 是一个list，包含K次增广图片batch
        而 un_targets 则只是一个 tensor，代表所有k次增广图片的标签
        """
        imgs = torch.cat((sup_imgs, *un_sup_imgs))
        targets = torch.cat([sup_targets, *[un_targets for _ in range(len(un_sup_imgs))]])
        return self.mixup(imgs, targets)

    def loss_ce_with_masked_(self,
                             logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor,
                             meter: Meter, name: str):
        meter[name] = (F.cross_entropy(logits, labels, reduction='none') * mask).mean()
        return meter[name]

    def loss_ce_with_targets_masked_(self,
                                     logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
                                     meter: Meter, name: str):
        meter[name] = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * targets, dim=1) * mask)
        return meter[name]


class PencilLoss(Loss):
    def loss_ce_with_lc_targets_(self, logits: torch.Tensor, targets: torch.Tensor, meter: Meter, name: str = 'Llc'):
        meter[name] = torch.mean(F.softmax(logits, dim=1) * (F.log_softmax(logits, dim=1) - torch.log((targets))))
        return meter[name]

    def loss_ent_(self, logits: torch.Tensor, meter: Meter, name='Lent'):
        meter[name] = - torch.mean(torch.mul(torch.softmax(logits, dim=1), torch.log_softmax(logits, dim=1)))
        return meter[name]


class FixMatchLoss(Loss):
    def loss_ce_with_masked_(self,
                             logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor,
                             meter: Meter, name: str):
        meter[name] = (F.cross_entropy(logits, labels, reduction='none') * mask).mean()
        return meter[name]

    def loss_ce_with_targets_masked_(self,
                                     logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
                                     meter: Meter, name: str):
        meter[name] = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * targets, dim=1) * mask)
        return meter[name]


class ICTLoss(Loss):
    def loss_mixup_sup_ce_(self, mixed_logits, labels_a, labels_b, lam, meter: Meter, name: str = 'Lsup'):
        loss = lam * F.cross_entropy(mixed_logits, labels_a) + (1 - lam) * F.cross_entropy(mixed_logits, labels_b)
        meter[name] = loss
        return loss

    def loss_mixup_unsup_mse_(self, input_logits, target_logits, decay, meter: Meter, name: str = 'Lunsup'):
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)
        loss = F.mse_loss(input_softmax, target_softmax)
        meter[name] = loss * decay
        return meter[name]

    def ict_mixup_(self, imgs: torch.Tensor, labels: torch.Tensor, mix=True):
        if mix:
            lam = np.random.beta(1.0, 1.0)
        else:
            lam = 1

        index = np.random.permutation(imgs.shape[0])

        mixed_x = lam * imgs + (1 - lam) * imgs[index, :]
        y_a, y_b = labels, labels[index]
        return mixed_x, y_a, y_b, lam

    def mixup_unsup_(self, imgs: torch.Tensor, logits: torch.Tensor, mix=True):
        '''
        Compute the mixup data. Return mixed inputs, mixed target, and lambda

        :param imgs:
        :param logits: 这里的preds 因为要混合，所以存在一个问题，是preds好还是logits好？
        :param mix:
        :return:
        '''
        if mix:
            lam = np.random.beta(mix, mix)
        else:
            lam = 1.

        index = torch.randperm(imgs.shape[0])
        mixed_x = lam * imgs + (1 - lam) * imgs[index, :]
        mixed_y = lam * logits + (1 - lam) * logits[index]

        return mixed_x, mixed_y, lam


class MentorLoss(Loss):
    def parse_dropout_rate_list_(self):
        """Parse a comma-separated string to a list.

        The format follows [dropout rate, epoch_num]+ and the result is a list of 100
        dropout rate.

        Args:
          str_list: the input string.
        Returns:
          result: the converted list
        """
        str_list = np.array([0.5, 17, 0.05, 78, 1.0, 5])
        values = str_list[np.arange(0, len(str_list), 2)]
        indexes = str_list[np.arange(1, len(str_list), 2)]

        values = [float(t) for t in values]
        indexes = [int(t) for t in indexes]

        assert len(values) == len(indexes) and np.sum(indexes) == 100
        for t in values:
            assert t >= 0.0 and t <= 1.0

        result = []
        for t in range(len(str_list) // 2):
            result.extend([values[t]] * indexes[t])
        return result

    def loss_ce_with_weight_(self):
        pass

    def mentor_mixup_(self, xs: torch.Tensor, targets: torch.Tensor,
                      weight: np.ndarray,
                      beta: float = 8.0):
        """
        MentorMix method.
        :param xs: the input image batch [batch_size, H, W, C]
        :param targets: the label batch  [batch_size, num_of_class]
        :param weight: mentornet weights
        :param beta: the parameter to sample the weight average.
        :return: The mixed images and label batches.
        """
        with torch.no_grad():
            if beta <= 0:
                return xs, targets

            idx = np.arange(xs.shape[0])
            idx = np.random.choice(idx, idx.shape[0], p=weight)
            xs_b = xs[idx]
            targets_b = targets[idx]

            mix = np.random.beta(beta, beta, xs.shape[0])
            mix = np.max([mix, 1 - mix], axis=0)
            mix[np.where(weight > 0.5)] = 1 - mix

            mixed_xs = xs * mix + xs_b * (1 - mix)
            mixed_targets = targets * mix[:, :, 0, 0] + targets_b * (1 - mix[:, :, 0, 0])
            return mixed_xs, mixed_targets


class IEGLossMixin(MixMatchMixin):
    def get_model(self):
        raise NotImplementedError()

    def create_metanet(self) -> Tuple[MetaModule, MetaSGD]:
        raise NotImplementedError()

    def loss_regularization_(self, model, l2_decay, meter: Meter, name: str):
        from thexp.contrib import ParamGrouper
        meter[name] = torch.sum(torch.pow(ParamGrouper(model).kernel_params(with_norm=False), 2)) * l2_decay
        return meter[name]

    def loss_softmax_cross_entropy_with_targets_(self, logits: torch.Tensor, targets: torch.Tensor, meter: Meter,
                                                 name: str):
        loss = -torch.mean(targets * torch.log_softmax(logits, dim=1))
        meter[name] = loss
        return meter[name]

    def weighted_loss_(self, logits: torch.Tensor, targets: torch.Tensor,
                       weighted: torch.Tensor,
                       meter: Meter, name: str):
        """带权重的损失"""
        loss_ = (targets * torch.log_softmax(logits, dim=1)) * weighted
        meter[name] = torch.sum(loss_)

        return meter[name]

    def _ieg_unsupvised_loss(self,
                             image, aug_image, val_image,
                             noisy_label, noisy_true_label, val_label,
                             meter: Meter):
        logits = self.to_logits(image)
        aug_logits = self.to_logits(aug_image)

        guess_targets = self.semi_mixmatch_loss(
            val_image, val_label,
            image, aug_image,
            un_logits=logits, un_aug_logits=aug_logits,
            un_true_labels=noisy_true_label, meter=meter)

        self.loss_kl_ieg_(logits, aug_logits, meter)  # + all_loss

        return logits, aug_logits, guess_targets

    def meta_optimize_(self,
                       noisy_images, noisy_labels, guess_targets,
                       clean_images, clean_labels,
                       meter: Meter):
        device = noisy_images.device
        batch_size = noisy_images.shape[0]
        metanet, metasgd = self.create_metanet()  # type: MetaModule,MetaSGD
        noisy_logits = metanet(noisy_images)
        noisy_targets = onehot(noisy_labels, guess_targets.shape[-1])
        eps_0 = torch.zeros([guess_targets.shape[0], 1],
                            dtype=torch.float,
                            device=device) + 0.9
        noisy_mixed_targets = eps_0 * noisy_targets + (1 - eps_0) * guess_targets
        noisy_loss = -torch.mean(noisy_mixed_targets * torch.log_softmax(noisy_logits, dim=1))

        weight_0 = torch.zeros(guess_targets.shape[0], dtype=torch.float, device=device) + (1 / batch_size)
        lookahead_loss = torch.sum(noisy_loss * weight_0) + self.loss_regularization_(self.model, 'l2_loss')

        val_grads = torch.autograd.grad(lookahead_loss, metanet.params())
        metasgd.meta_step(val_grads)

        val_logits = metanet(clean_images)
        val_targets = onehot(clean_labels, val_logits.shape[-1])
        val_meta_loss = -torch.mean(torch.sum(
            F.log_softmax(val_logits, dim=1) * val_targets,
            dim=1)) + self.loss_regularization_(metanet, meter, 'metal2_loss')

        meta_grad = torch.autograd.grad(val_meta_loss, metanet.params())
        weight_grad, eps_grad = torch.autograd.grad(metanet.params(), [weight_0, eps_0], meta_grad)

        # weight_0 - weight_grad - (1/batchsize)
        weight_1 = torch.clamp_min(weight_0 - weight_grad - (1 / batch_size), 0)
        weight_1 = weight_1 / (torch.sum(weight_1) + 1e-5)

        weight_1 = weight_1.detach()
        eps_1 = (eps_grad < 0).float().detach()

        return weight_1, eps_1, noisy_targets

    def loss_kl_ieg_(self, q_logits, p_logits, consistency_factor, meter: Meter, name='Lkl'):
        q = torch.softmax(q_logits, dim=1)
        per_example_kl_loss = q * (
                torch.log_softmax(q_logits, dim=1) - torch.log_softmax(p_logits, dim=1))
        meter[name] = per_example_kl_loss.mean() * q.shape[-1] * consistency_factor
        return meter[name]

    def noisy_ieg_loss_(self,
                        val_images: torch.Tensor,
                        val_label: torch.Tensor,
                        noisy_images: torch.Tensor, noisy_aug_images: torch.Tensor,
                        noisy_labels: torch.Tensor,
                        noisy_true_labels: torch.Tensor,
                        meter: Meter):
        # meter.loss =
        noisy_logits = self.to_logits(noisy_images)

        # other losses TODO
        # 三个损失，MixMatch的两个+一个KL散度损失
        logits, aug_logits, guess_targets = self._ieg_unsupvised_loss(noisy_images, noisy_aug_images, val_images,
                                                                      noisy_labels, noisy_true_labels, val_label,
                                                                      meter=meter)

        weight_1, eps_1, noisy_targets = self.meta_optimize_(noisy_images, noisy_labels, guess_targets,
                                                             val_images, val_label, meter)

        mixed_targets = eps_1 * noisy_targets + (1 - eps_1) * guess_targets
        net_loss1 = self.loss_softmax_cross_entropy_with_targets_(mixed_targets, logits,
                                                                  meter=meter, name='net_loss')

        init_mixed_labels = 0.9 * noisy_targets + 0.1 * guess_targets
        net_loss2 = self.weighted_loss_(noisy_logits, init_mixed_labels,
                                        weight_1,
                                        meter=meter, name='init_net_loss')

        meter.all_loss = meter.all_loss + (net_loss1 + net_loss2) / 2
        meter.all_loss = meter.all_loss + self.loss_regularization_(
            self.get_model(),
            meter=meter, name='l2_loss')
