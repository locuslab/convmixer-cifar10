Train ConvMixer on CIFAR-10
-----------------------------------
 ✈️ 🚗 🐦 🐈 🦌 🐕 🐸 🐎 🚢 🚚
 

This is a simple ConvMixer training script for CIFAR-10. It's probably a good starting point for new experiments on small datasets.

For training on ImageNet and/or reproducing our original results, see the [main ConvMixer repo](https://github.com/locuslab/convmixer).

You can get around **92.5%** accuracy in just **25 epochs** by running the script with the following arguments,
which trains a ConvMixer-256/8 with kernel size 5 and patch size 2.

```
python train.py --lr-max=0.05 --ra-n=2 --ra-m=12 --wd=0.005 --scale=1.0 --jitter=0 --reprob=0
```

Here's an example of the output of the above command (on a 2080Ti GPU):

```
[ConvMixer] Epoch: 0  | Train Acc: 0.3938, Test Acc: 0.5418, Time: 43.2, lr: 0.005000
[ConvMixer] Epoch: 1  | Train Acc: 0.6178, Test Acc: 0.6157, Time: 42.6, lr: 0.010000
[ConvMixer] Epoch: 2  | Train Acc: 0.7012, Test Acc: 0.7069, Time: 42.6, lr: 0.015000
[ConvMixer] Epoch: 3  | Train Acc: 0.7383, Test Acc: 0.7708, Time: 42.7, lr: 0.020000
[ConvMixer] Epoch: 4  | Train Acc: 0.7662, Test Acc: 0.7344, Time: 42.5, lr: 0.025000
[ConvMixer] Epoch: 5  | Train Acc: 0.7751, Test Acc: 0.7655, Time: 42.4, lr: 0.030000
[ConvMixer] Epoch: 6  | Train Acc: 0.7901, Test Acc: 0.8328, Time: 42.6, lr: 0.035000
[ConvMixer] Epoch: 7  | Train Acc: 0.7974, Test Acc: 0.7655, Time: 42.4, lr: 0.040000
[ConvMixer] Epoch: 8  | Train Acc: 0.8040, Test Acc: 0.8138, Time: 42.6, lr: 0.045000
[ConvMixer] Epoch: 9  | Train Acc: 0.8084, Test Acc: 0.7891, Time: 42.5, lr: 0.050000
[ConvMixer] Epoch: 10 | Train Acc: 0.8237, Test Acc: 0.8387, Time: 42.8, lr: 0.045250
[ConvMixer] Epoch: 11 | Train Acc: 0.8373, Test Acc: 0.8312, Time: 42.6, lr: 0.040500
[ConvMixer] Epoch: 12 | Train Acc: 0.8529, Test Acc: 0.8563, Time: 42.5, lr: 0.035750
[ConvMixer] Epoch: 13 | Train Acc: 0.8657, Test Acc: 0.8700, Time: 42.7, lr: 0.031000
[ConvMixer] Epoch: 14 | Train Acc: 0.8751, Test Acc: 0.8527, Time: 42.6, lr: 0.026250
[ConvMixer] Epoch: 15 | Train Acc: 0.8872, Test Acc: 0.8907, Time: 42.5, lr: 0.021500
[ConvMixer] Epoch: 16 | Train Acc: 0.8979, Test Acc: 0.9019, Time: 42.7, lr: 0.016750
[ConvMixer] Epoch: 17 | Train Acc: 0.9080, Test Acc: 0.9068, Time: 42.9, lr: 0.012000
[ConvMixer] Epoch: 18 | Train Acc: 0.9198, Test Acc: 0.9139, Time: 42.5, lr: 0.007250
[ConvMixer] Epoch: 19 | Train Acc: 0.9316, Test Acc: 0.9240, Time: 42.6, lr: 0.002500
[ConvMixer] Epoch: 20 | Train Acc: 0.9383, Test Acc: 0.9238, Time: 42.8, lr: 0.002000
[ConvMixer] Epoch: 21 | Train Acc: 0.9407, Test Acc: 0.9248, Time: 42.5, lr: 0.001500
[ConvMixer] Epoch: 22 | Train Acc: 0.9427, Test Acc: 0.9253, Time: 42.6, lr: 0.001000
[ConvMixer] Epoch: 23 | Train Acc: 0.9445, Test Acc: 0.9255, Time: 42.5, lr: 0.000500
[ConvMixer] Epoch: 24 | Train Acc: 0.9441, Test Acc: 0.9260, Time: 42.6, lr: 0.000000
```

By adding more regularization (data augmentation) and training for four times longer, you can get **more than 94% accuracy**:

```
python train.py --lr-max=0.05 --ra-n=2 --ra-m=12 --wd=0.005 --scale=1.0 --jitter=0.2 --reprob=0.2 --epochs=100
```


Note that this script is not intended to perfectly replicate the results in our paper, as PyTorch's built-in data augmentation methods (for RandAugment and Random Erasing in particular) differ slightly from those of the library we used, [pytorch-image-models](https://github.com/rwightman/pytorch-image-models). This script also does not include Mixup/Cutmix, as these are not provided by PyTorch (torchvision.transforms) and we wanted to keep it as simple as possible. That said, you can probably get similar results by experimenting with different amounts of regularization with this script.

Feel free to open an issue if you have any questions.
