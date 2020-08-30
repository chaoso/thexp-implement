from torchvision.transforms import ToTensor
from torchvision import transforms
from thexp.contrib.data.augments.image import RandAugmentMC


class Weak(object):
    def __init__(self, mean=None, std=None):
        lis = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=4,
                                  padding_mode='reflect'),
            transforms.ToTensor(),

        ]
        if mean is not None and std is not None:
            lis.append(transforms.Normalize(mean=mean, std=std))
        self.weak = transforms.Compose(lis)

    def __call__(self, x):
        return self.weak(x)


class Strong():
    def __init__(self, mean=None, std=None) -> None:
        lis = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=4,
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
        ]
        if mean is not None and std is not None:
            lis.append(transforms.Normalize(mean=mean, std=std))
        self.weak = transforms.Compose(lis)
        self.strong = transforms.Compose(lis)

    def __call__(self, x):
        return self.strong(x)


class ToNormTensor():
    def __init__(self, mean=None, std=None):
        if mean is not None and std is not None:
            self.norm = transforms.Normalize(mean=mean, std=std)
        else:
            self.norm = None
        self.totensor = transforms.ToTensor()

    def __call__(self, x):
        val = self.totensor(x)
        if self.norm is not None:
            return self.norm(val)
        else:
            return val
