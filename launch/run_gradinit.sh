#!/usr/bin/env bash

function runexp {

gpu=${1}
arch=${2}
alg=${3}
glr=${4}
iters=${5}
flags=${6}

flags_print="$(echo -e "${flags}" | tr -d '[:space:]')"
flags_print=${flags_print//--/_}

expname=gradinit-${arch}-cifar10-cutout-alg_${alg}-glr_${glr}-i_${iters}-sgclip_${flags_print}

cmd="
CUDA_VISIBLE_DEVICES=${gpu}
python train_cifar.py  --arch ${arch} --cutout
    --gradinit  --gradinit-alg ${alg} --gradinit-eta 0.1
    --gradinit-gamma 1 --gradinit-normalize-grad
    --gradinit-lr ${glr}  --gradinit-min-scale 0.01
    --gradinit-iters ${iters} --gradinit-grad-clip 1
    --expname ${expname} ${flags}
"

eval ${cmd}

}

# runexp  gpu    arch    alg   glr   iters    flags
runexp     0    vgg19    sgd   1e-2    390    "--no_bn --train-clip 1 --seed 4096"
runexp     0    vgg19    sgd   1e-1    390    "--seed 4096"
#
runexp     0 resnet110   sgd   5e-2    390    "--no_bn --train-clip 1 --seed 4096"
runexp     0 resnet110   sgd   1e-3    390    "--seed 4096"
