"""
som constant values
"""

# value from https://github.com/kekmodel/FixMatch-pytorch/blob/master/dataset/cifar.py
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

# value from https://github.com/danieltan07/learning-to-reweight-examples/blob/master/data_loader.py
mnist_mean = (0.1307,)
mnist_std = (0.3081,)

# default value
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

norm_val = {
    'cifar10': dict(mean=cifar10_mean, std=cifar10_std),
    'cifar100': dict(mean=cifar100_mean, std=cifar100_std),
    'mnist': dict(mean=mnist_mean, std=mnist_std),
    'default': dict(mean=normal_mean, std=normal_std),
    'none': dict(mean=None, std=None),
}
