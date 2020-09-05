from thexp import Trainer
import numpy as np
from trainers import *
from thexp.contrib.data import splits
from thexp import DatasetBuilder, DataBundler

from data.constant import norm_val
from data.transforms import ToTensor
from data.dataxy import datasets
from data.transforms import Weak, Strong, ToNormTensor


class DatasetMixin(Trainer):
    def datasets(self, params: GlobalParams):
        raise NotImplementedError()


class Base32Mixin(Trainer):
    """
    all 32*32 dataset, including cifar10, cifar100, svhn

    use base train data shape: ids, xs, aug_xs, ys

    test data shape: xs, ys
    """

    def datasets(self, params: GlobalParams):
        dataset_fn = datasets[params.dataset]

        test_x, testy = dataset_fn(False)
        train_x, train_y = dataset_fn(True)

        train_idx, val_idx = splits.train_val_split(train_y, val_size=params.val_size)

        mean, std = norm_val.get(params.dataset, [None, None])
        toTensor = ToNormTensor(mean, std)
        weak = Weak(mean, std)
        strong = Strong(mean, std)

        test_dataloader = (
            DatasetBuilder(test_x, testy)
                .add_x(transform=toTensor)
                .add_y()
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers)
        )

        train_dataloader = (
            DatasetBuilder(train_x, train_y)
                .toggle_id()
                .add_x(transform=weak)
                .add_x(transform=strong)
                .add_y()
                .subset(train_idx)
                .DataLoader(batch_size=params.batch_size,
                            num_workers=params.num_workers,
                            shuffle=True)
        )

        val_datalaoder = (
            DatasetBuilder(train_x, train_y)
                .add_x(transform=toTensor).add_y()
                .subset(val_idx)
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers)
        )

        self.regist_databundler(train=train_dataloader,
                                eval=val_datalaoder,
                                test=test_dataloader)

        self.to(self.device)


class SyntheticNoisyMixin(Trainer):
    """
    supervised image dataset with synthetic noisy

    train: (ids, xs, axs, ys, nys)
    val: xs, ys
    test: xs, ys

    Note:
        the first 'ys' right label and is used to calculate accuracy, and will not be used to train model.
    """

    def datasets(self, params: GlobalParams):
        params.noisy_type = params.default('symmetric', True)
        params.noisy_ratio = params.default(0.2, True)

        import numpy as np
        dataset_fn = datasets[params.dataset]
        test_x, test_y = dataset_fn(False)
        train_x, train_y = dataset_fn(True)

        train_ids, val_ids = splits.train_val_split(train_y, val_size=5000)

        train_x, val_x = train_x[train_ids], train_x[val_ids]
        train_y, val_y = train_y[train_ids], train_y[val_ids]

        mean, std = norm_val.get(params.dataset, [None, None])
        toTensor = ToNormTensor(mean, std)
        weak = Weak(mean, std)
        strong = Strong(mean, std)

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

        train_set = (
            DatasetBuilder(train_x, train_y)
                .add_labels(noisy_y, 'noisy_y')
                .toggle_id()
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

        self.regist_databundler(train=train_dataloader,
                                eval=val_dataloader,
                                test=test_dataloader)
        self.to(self.device)


class FixMatchDatasetMixin(Trainer):
    """semi-supervised image dataset for fixmatch"""

    def datasets(self, params: SemiSupervisedParams):
        dataset_fn = datasets[params.dataset]

        test_x, test_y = dataset_fn(False)
        train_x, train_y = dataset_fn(True)

        indexs, un_indexs, val_indexs = splits.semi_split(train_y, n_percls=params.n_percls, val_size=5000,
                                                          repeat_sup=False)

        mean, std = norm_val.get(params.dataset, [None, None])
        toTensor = ToNormTensor(mean, std)
        weak = Weak(mean, std)
        strong = Strong(mean, std)

        sup_set = (
            DatasetBuilder(train_x, train_y)
                .add_x(transform=weak)
                .add_y()
                .subset(indexs)
        )

        unsup_set = (
            DatasetBuilder(train_x, train_y)
                .toggle_id()
                .add_x(transform=weak)
                .add_x(transform=strong)
                .add_y()
                .subset(un_indexs)
        )
        self.cl_set = unsup_set

        sup_dataloader = sup_set.DataLoader(batch_size=params.batch_size, num_workers=params.num_workers,
                                            shuffle=True)
        self.sup_dataloader = sup_dataloader

        unsup_dataloader = unsup_set.DataLoader(batch_size=params.batch_size * params.uratio,
                                                num_workers=params.num_workers,
                                                shuffle=True)

        self.unsup_dataloader = DataBundler().add(unsup_dataloader).to(self.device)

        val_dataloader = (
            DatasetBuilder(train_x[val_indexs], train_y[val_indexs])
                .add_x(transform=toTensor).add_y()
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True)
        )

        test_dataloader = (
            DatasetBuilder(test_x, test_y)
                .add_x(transform=toTensor).add_y()
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True)
        )

        self.regist_databundler(train=DataBundler().add(sup_dataloader).add(unsup_dataloader).zip_mode(),
                                eval=val_dataloader,
                                test=test_dataloader)
        self.to(self.device)


class MixMatchDatasetMixin(Trainer):
    """semi-supervised image dataset for fixmatch"""

    def datasets(self, params: SemiSupervisedParams):
        dataset_fn = datasets[params.dataset]

        test_x, test_y = dataset_fn(False)
        train_x, train_y = dataset_fn(True)

        indexs, un_indexs, val_indexs = splits.semi_split(train_y, n_percls=params.n_percls, val_size=5000,
                                                          repeat_sup=False)

        mean, std = norm_val.get(params.dataset, [None, None])
        toTensor = ToNormTensor(mean, std)
        weak = Weak(mean, std)

        sup_set = (
            DatasetBuilder(train_x, train_y)
                .add_x(transform=weak)
                .add_y()
                .subset(indexs)
        )

        params.K = params.default(2, True)
        unsup_set = DatasetBuilder(train_x, train_y)
        for _ in range(params.K):
            unsup_set.add_x(transform=weak)
        unsup_set = unsup_set.add_y().subset(un_indexs)

        sup_dataloader = sup_set.DataLoader(batch_size=params.batch_size,
                                            num_workers=params.num_workers,
                                            shuffle=True)

        unsup_dataloader = unsup_set.DataLoader(batch_size=params.batch_size,
                                                num_workers=params.num_workers,
                                                shuffle=True)

        val_dataloader = (
            DatasetBuilder(train_x[val_indexs], train_y[val_indexs])
                .add_x(transform=toTensor).add_y()
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True)
        )

        test_dataloader = (
            DatasetBuilder(test_x, test_y)
                .add_x(transform=toTensor).add_y()
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True)
        )

        self.regist_databundler(train=DataBundler().cycle(sup_dataloader).add(unsup_dataloader).zip_mode(),
                                eval=val_dataloader,
                                test=test_dataloader)
        self.to(self.device)


class MnistImblanceDataset(Trainer):
    def datasets(self, params: MnistImblanceParams):
        super().datasets(params)
        from data.dataxy import mnist

        test_x, test_y = mnist(False)
        train_x, train_y = mnist(True)

        train_y = np.array(train_y, dtype=np.float32)
        test_y = np.array(test_y, dtype=np.float32)

        # search and mask sample with class [4, 9]
        train_mask_lis = [np.where(train_y == i)[0] for i in params.train_classes]
        test_mask_lis = [np.where(test_y == i)[0] for i in params.train_classes]

        for new_cls, i in enumerate(params.train_classes):
            train_y[train_mask_lis[new_cls]] = new_cls
            test_y[test_mask_lis[new_cls]] = new_cls

        train_mask = np.concatenate(train_mask_lis)
        test_mask = np.concatenate(test_mask_lis)

        test_x, test_y = test_x[test_mask], test_y[test_mask]
        train_x, train_y = train_x[train_mask], train_y[train_mask]

        # split train/val dataset
        train_ids, val_ids = splits.train_val_split(train_y, val_size=params.val_size)

        train_x, val_x = train_x[train_ids], train_x[val_ids]
        train_y, val_y = train_y[train_ids], train_y[val_ids]

        # reduce size of second class
        train_mask_lis = [np.where(train_y == i)[0] for i in range(len(params.train_classes))]
        sec_cls_size = int((1 - params.train_proportion) * len(train_mask_lis[0]))
        train_mask_lis[1] = train_mask_lis[1][:sec_cls_size]
        train_mask = np.concatenate(train_mask_lis)
        train_x, train_y = train_x[train_mask], train_y[train_mask]

        toTensor = ToNormTensor((0.1307,), (0.3081,))

        train_dataloader = (
            DatasetBuilder(train_x, train_y)
                .add_x(toTensor).add_y()
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True)
        )

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

        self.regist_databundler(
            train=DataBundler().add(train_dataloader).cycle(val_dataloader).zip_mode(),
            eval=val_dataloader,
            test=test_dataloader)
        self.to(self.device)
