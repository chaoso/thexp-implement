from thexp import Params


class GlobalParams(Params):

    def __init__(self):
        super().__init__()
        self.epoch = 400
        self.optim = self.create_optim('SGD',
                                       lr=0.05,
                                       momentum=0.9,
                                       weight_decay=5e-4,
                                       nesterov=True)

        self.eval_test_per_epoch = (5, 10)

        self.dataset = self.choice('dataset', 'cifar10', 'cifar100', 'mnist', 'fashionmnist', 'svhn')
        self.n_classes = 10
        self.topk = (1, 2, 3, 4)

        self.batch_size = 64
        self.num_workers = 4

        self.ema = True
        self.ema_alpha = 0.999

        self.val_size = 5000

        self.architecture = self.choice('architecture', 'WRN', 'Resnet')
        self.depth = 28  # for both wideresnet and resnet
        self.widen_factor = 2  # for wideresnet

    def wideresnet282(self):
        """model for semisupervised"""
        self.architecture = 'WRN'
        self.depth = 28
        self.widen_factor = 2

    def resnet32(self):
        """model for noisy label"""
        self.architecture = 'Resnet'
        self.depth = 32

    def initial(self):
        if self.dataset in {'cifar100'}:
            self.n_classes = 100
        if self.ENV.IS_PYCHARM_DEBUG:
            self.num_workers = 0

        self.lr_sche = self.SCHE.Cos(start=self.optim.args.lr,
                                     end=0.00001,
                                     right=self.epoch)


class SupervisedParams(GlobalParams):
    def __init__(self):
        super().__init__()
        self.epoch = 600
        self.batch_size = 128
        self.optim = self.create_optim('SGD',
                                       lr=0.3,
                                       momentum=0.9,
                                       weight_decay=5e-4,
                                       nesterov=True)

    def iter_baseline(self):
        self.architecture = 'WRN'
        for p in self.grid_search('dataset', ['cifar10', 'cifar100']):
            p.depth = 28
            for pp in p.grid_search('widen_factor', [2, 10]):
                yield pp
        self.architecture = 'Resnet'
        for p in self.grid_search('dataset', ['cifar10', 'cifar100']):
            for pp in p.grid_search('depth', [20, 32, 44, 50, 56]):
                yield pp


class SemiSupervisedParams(GlobalParams):
    def __init__(self):
        super().__init__()
        self.epoch = 2048
        self.batch_size = 64
        self.optim = self.create_optim('SGD',
                                       lr=0.03,
                                       momentum=0.9,
                                       weight_decay=5e-4,
                                       nesterov=True)

        self.uratio = 7
        self.pred_thresh = 0.95
        self.n_percls = 400

    def iter_baseline(self):
        self.architecture = 'WRN'
        self.depth = 28
        self.widen_factor = 2
        for p in self.grid_search('dataset', ['cifar10', 'cifar100', 'svhn']):
            yield p


class NoisyParams(GlobalParams):
    def __init__(self):
        super().__init__()
        self.epoch = 600
        self.batch_size = 128

        self.noisy_ratio = 0.25
        self.noisy_type = self.choice('noisy_type', 'symmetric', 'asymmetric')

        self.optim = self.create_optim('SGD',
                                       lr=0.3,
                                       momentum=0.9,
                                       weight_decay=5e-4,
                                       nesterov=True)


class MnistImblanceParams(GlobalParams):

    def __init__(self):
        super().__init__()

        self.optim = self.create_optim('SGD',
                                       lr=1e-3,
                                       momentum=0.9,
                                       weight_decay=5e-4,
                                       nesterov=True)
        self.epoch = 25  # is enough for mnist
        self.batch_size = 100
        self.train_classes = [9, 4]
        self.train_proportion = 0.995
        self.val_size = 1000
        self.ema = False
