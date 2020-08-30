from typing import Dict, Callable, Tuple

from PIL import Image
from thexp import globs
from thexp.decorators import regist_func
from thexp.base_classes import llist
from torchvision.datasets.cifar import CIFAR10, CIFAR100
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torchvision.datasets.svhn import SVHN

# globs.add_value('datasets', 'path/to/all_datasets/', level=globs.LEVEL.globals)
root = globs['datasets']

datasets = {
    # 'cifar10': cifar10,
}  # type:Dict[str,Callable[[bool],Tuple[llist,llist]]]


@regist_func(datasets)
def cifar10(train=True):
    dataset = CIFAR10(root=root, train=train)
    xs = llist(Image.fromarray(i) for i in dataset.data)
    ys = llist(int(i) for i in dataset.targets)

    return xs, ys


@regist_func(datasets)
def cifar100(train=True):
    dataset = CIFAR100(root=root, train=train)
    xs = llist(Image.fromarray(i) for i in dataset.data)
    ys = llist(int(i) for i in dataset.targets)

    return xs, ys


@regist_func(datasets)
def svhn(train=True):
    dataset = SVHN(root=root, split='train' if train else 'test')
    xs = llist(Image.fromarray(np.transpose(img, (1, 2, 0))) for img in dataset.data)
    ys = llist(int(i) for i in dataset.labels)

    return xs, ys


@regist_func(datasets)
def mnist(train=True):
    dataset = MNIST(root=root, train=train)
    xs = llist(Image.fromarray(np.stack([img.numpy()] * 3, axis=2)) for img in dataset.data)
    ys = llist(int(i) for i in dataset.targets)

    return xs, ys


@regist_func(datasets)
def fashionmnist(train=True):
    dataset = FashionMNIST(root=root, train=train)
    xs = llist(Image.fromarray(np.stack([img.numpy()] * 3, axis=2)) for img in dataset.data)
    ys = llist(int(i) for i in dataset.targets)

    return xs, ys
