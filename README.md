# GradInit
This repository hosts the code for experiments in the paper, [GradInit: Learning to Initialize Neural Networks for Stable and Efficient Training](http://arxiv.org/abs/2102.08098). 
 
Scripts for experiments on CIFAR-10 is currently available. Please refer to `launch/run_gradinit_densenet.sh` for DenseNet-100, `launch/run_gradinit_wrn.sh` for WRN-28-10, and `launch/run_gradinit.sh` for other networks shown in the paper. We will release the code for ImageNet and IWSLT experiments soon. 

## Notes
**May 24, 2022**: Releasing the [code for IWSLT'14](tree/master/fairseq). Code of the whole fairseq library is inlucded, where we only modified `fairseq/dataclass/configs.py` to add configurations for GradInit without causing import order conflicts. The implementation of GradInit is under `fairseq/gradinit`. 

**Feb 17, 2021**: Releasing the code for training CNNs on CIFAR-10.

**March 9, 2021**: Update the code to support any architecture with only `nn.Conv2d`, `nn.Linear` and `nn.BatchNorm2d` as the parameterized layers. Simply call `gradinit_utils.gradinit` before your training loop. Further extensions to other parameterized layers can be achieved by modifying `gradinit_utils.gradinit.get_ordered_params`, `gradinit_utils.take_opt_step` and `gradinit_utils.gradinit.recover_params` to iterate over all parameters of these layers.