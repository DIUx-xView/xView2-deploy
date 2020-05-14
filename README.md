# Dual-HRNet for Building Localization and DamageClassification

## Abstact
Comparison of pre-disaster and post-disaster on satellites imagery play an important roles for the humanitarian assistance and disaster relief (HADR) efforts while severe global events. Automatic comparison using AI helps identify the affected area. In this study, we propose Dual-HRNets for both localizing buildings and classifying their damage level on the satellite imagery simultaneously. Since the goal of this study is to predict the location of buildings on the pre-disaster image as well as classify the damage level in accordance with the post-disaster image, our Dual-HRNets take a pair of pre- and post-disaster images over each HRNet. Each intermediate stage of Dual-HRNet, the features of each HRNet are fused by a fusion block. Our experiments present a good performance of Dual-HRNets for the task of both localization and classification. Our approach with Dual-HRNet achieved the 5th place in xView2 challenge over 3,500 participants.

![](figures/dual-hrnet.png)

## Qualitative results of Dual-HRNet
From left to right, the figures shows pre-disaster images, results of Dual-HRnet and post-disaster images.
White, yellow, orange and red regions represent nodamage, minor damage, major damage and destroyed respectively.
![](figures/test_1.png)
![](figures/test_2.png)
![](figures/test_3.png)


## Quick start
### Install
Install PyTorch=1.4 following the [official instructions](https://pytorch.org/).

Please refer to [submission/Dockerfile](submission/Dockerfile)

### Train and test
Please batch size in configuration file(dual-hrnet.yaml) for your environment.

Train
````bash
python train_net.py --config_file configs/dual-hrnet.yaml \
                    --ckpt_save_dir=PATH_TO_SAVE_CHECKPOINTS
````

Test
````bash
python test_net.py --config_file configs/dual-hrnet.yaml \
                   --ckpt_path=PATH_TO_LOAD_CHECKPOINTS
````

## License
Dual-HRNet is released under the [MIT license](LICENSE).

## Acknowledgement
We modified code based on HRNet Semantic Segmentation(https://github.com/HRNet/HRNet-Semantic-Segmentation)
