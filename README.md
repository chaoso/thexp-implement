# Thexp-implement

Reimplement of some papers I interested. All implementations are based on my another pytorch framework [thexp](https://github.com/sailist/thexp), and are as elegant and simple as I can.

## How to run
```
git clone https://github.com/sailist/thexp-implement
```

You need to instal my another library [thexp](https://github.com/sailist/thexp).
```
pip install thexp
```

When first run, you need to specify the dataset root: open `data/dataxy.py`, and change the value of the variable `root`. That's the only thing you need to do before running some scripts.  

And, find the script in `trainers/` and directly run it.

For example, if you want to reproduce the result of fixmatch, you can find it in `trainers/semisupervised/fixmatch.py`, and run it by using the code below:
```
python trainers/semisupervised/fixmatch.py

# or
cd trainers/semisupervised
python fixmatch.py
``` 

> the working directory will be changed automaticaly, so you just need to run it.

## Implementation list

Here list the paper reimplemented in this repo.

### Supervised baseline
including WRN-28-2, WRN-28-10, and Resnet{20, 32, 44, 50, 56},

Only use basic cross-entropy loss to optimize the model, and the data use four common augmentation methods:weak, weak+mixup, strong, strong+mixup

> `weak` means random horizontal flip and random crop, and `strong` means `weak` + `RandAugment` , please see the code for details.



For most case, Strong or Strong+mixup is the best 

> [1] RandAugment: Practical automated data augmentation
  with a reduced search space, https://arxiv.org/pdf/1909.13719.pdf
>
> [2] mixup: Beyond Empirical Risk Minimization, https://arxiv.org/abs/1710.09412 

### Semi-Supervised
#### Interpolation Consistency Training for Semi-Supervised Learning
 [[paper]](https://arxiv.org/abs/1903.03825)

```
python3 trainers/semisupervised/ict.py
```
 
#### MixMatch: A Holistic Approach to Semi-Supervised Learning
 [[paper]](https://arxiv.org/abs/1905.02249)

```
python3 trainers/semisupervised/mixmatch.py
```

#### FixMatch: Simplifying Semi-Supervised Learning with Consistency and Conﬁdence
 [[paper]](https://arxiv.org/abs/2001.07685)
```
python3 trainers/semisupervised/fixmatch.py
```

### Noisy Label
#### MentorNet: Learning Data-Driven Curriculum for Very Deep Neural Networks

[[paper]](https://arxiv.org/abs/1712.05055)
 - [ ] Todo 
 
 
#### Probabilistic End-to-end Noise Correction for Learning with Noisy Labels(Pencil)
[[paper]](https://arxiv.org/abs/1903.07788) 

```
python3 trainers/noisylabel/pencil.py
```
 
#### Beyond Synthetic Noise: Deep Learning on Controlled Noisy Labels (MentorMix)
[[paper]](https://arxiv.org/abs/1911.09781)

 - [ ] Todo
 
### Meta-learning

#### Learning to Reweight Examples for Robust Deep Learning (L2R)
[[paper]](https://arxiv.org/abs/1803.09050)

reproduce the result on class imblance, and can't reach the result on noisy label experiment.

```
python3 trainers/metalearning/l2r_imblance.py
```

#### Distilling Effective Supervision from Severe Label Noise (IEG)

> Another name —— IEG: Robust neural net training with severe label noises 

 [[paper]](https://arxiv.org/abs/1910.00701)
 
use wideresnet-28-2, and reproduce the result on cifar10 with 40%, 80% synthetic noisy.(I can't run wideresnet-28-10 on my single GPU.) 
 
 **cifar10** 
 
 |noisy ratio|results|
 |---|---|
 |0.4|95.01|
 |0.8|94.81|
 
> reach cifar100 results may need training larger model(like WRN28-10 or others) 

```
python3 trainers/metalearning/ieg_noisy_label.py --noisy_ratio=0.8
```
