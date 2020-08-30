# Thexp-implement

Reimplement of some papers I interested. All implementations are based on my another pytorch framework [thexp](https://github.com/sailist/thexp), and  are elegant and simple as I can.

## How to run
```
git clone https://github.com/sailist/thexp-implement
```

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

## implementation list

Here list the paper reimplemented in this repo.
### Supervised baseline
including WRN-28-2, WRN-28-10, and Resnet{20, 32, 44, 50, 56},

Only use basic cross-entropy loss to optimize the model, and the data use four common augmentation methods:weak, weak+mixup, strong, strong+mixup

> `weak` means random horizontal flip and random crop, and `strong` means `weak` + `RandAugment` , please see the code for details.

> [1] RandAugment: Practical automated data augmentation
  with a reduced search space, https://arxiv.org/pdf/1909.13719.pdf

### Semi-Supervised
 - Interpolation Consistency Training for Semi-Supervised Learning, https://arxiv.org/abs/1903.03825
 - MixMatch: A Holistic Approach to Semi-Supervised Learning, https://arxiv.org/abs/1905.02249
 - FixMatch: Simplifying Semi-Supervised Learning with Consistency and Conﬁdence, https://arxiv.org/abs/2001.07685


### Noisy Label
 
 - [ ] MentorNet: Learning Data-Driven Curriculum for Very Deep Neural Networks, https://arxiv.org/abs/1712.05055
 - Probabilistic End-to-end Noise Correction for Learning with Noisy Labels, https://arxiv.org/abs/1903.07788
 - [ ] Beyond Synthetic Noise: Deep Learning on Controlled Noisy Labels, https://arxiv.org/abs/1911.09781
 